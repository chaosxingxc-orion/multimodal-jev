"""Evaluate one audio/candidate pair at a time with the training card layout."""
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path

import torch

from multimodal_jev.audio_policy import load_audio
from multimodal_jev.baseline import file_hash
from multimodal_jev.context_decision_contract import encode_context_request
from multimodal_jev.single_card_forward import request
from multimodal_jev.stage1_layout_cache import split_text
from tools.reports.card_scoring_benchmark import batches, render


def read_pinned(pin):
    path = Path(pin['path'])
    if file_hash(path) != pin['sha256']:
        raise ValueError(f'changed evaluation input: {path.name}')
    return json.loads(path.read_bytes())


def bounded_batches(rows, width, chunk_size, max_chunks):
    """Respect both the pair limit and the padded audio work of each forward."""
    pending = {}
    for record in rows:
        for item in record:
            inputs = item['inputs']
            bucket = min(int(inputs['input_features_mask'].sum()), chunk_size)
            chunks = (inputs['input_features'].shape[-1] + chunk_size - 1) // chunk_size
            if not 0 < chunks <= max_chunks:
                raise ValueError('audio exceeds the declared single-pair chunk budget')
            group, largest = pending.setdefault(bucket, ([], 0))
            if group and (len(group) >= width or (len(group) + 1) * max(largest, chunks) > max_chunks):
                yield group
                group, largest = [], 0
            group.append(item)
            pending[bucket] = (group, max(largest, chunks))
    for group, _ in pending.values():
        if group:
            yield group


class SingleCardScorer:
    def __init__(self, spec, processor, candidates, collate, *, requests=None):
        frozen = read_pinned(spec['cards'])
        self.candidates = list(candidates)
        self.collate = collate
        self.width = spec['pair_batch']
        self.max_tokens = spec['encoding']['max_tokens']
        self.max_chunks = spec.get('max_batch_audio_chunks')
        self.parity_pairs = spec.get('parity_pairs', 1)
        if (self.max_chunks is not None and (type(self.max_chunks) is not int or self.max_chunks < 1)
                or type(self.parity_pairs) is not int or self.parity_pairs < 1):
            raise ValueError('positive audio work limit and parity count required')
        if type(self.width) is not int or self.width < 2 or not self.candidates:
            raise ValueError('explicit pair batch size and ordered candidates required')
        self.requests = requests
        if requests is None:
            cards = {frozen['catalog'][index]['id']: frozen['catalog'][index]
                     for index in frozen['arms']['rules']}
            self.source_tiles = tuple(tuple(torch.tensor(cards[key][part]) for part in ('prefix', 'suffix'))
                                      for key in self.candidates)
        else:
            witness = spec['witness']
            if file_hash(witness['path']) != witness['sha256']:
                raise ValueError('single-card encoding witness changed')
            audio = load_audio(witness['path'], None)
            encoding = spec['encoding']

            @lru_cache(maxsize=spec['token_cache_entries'])
            def compile_card(question, requirements):
                card = request(dict(id=question.id, definition=question.definition),
                               requirements, frozen['config'])
                card = replace(card, questions=(question,))
                encoded, positions = encode_context_request(processor, card, audio, **encoding)
                prefix, suffix, _ = split_text(encoded['input_ids'][0], encoding['audio_token_id'])
                if positions.numel() != 1:
                    raise ValueError('exactly one candidate readout required')
                return torch.tensor(prefix), torch.tensor(suffix)

            self.compile_card = compile_card

    def tiles(self, uid):
        if self.requests is None:
            return self.source_tiles
        source = self.requests[uid]
        if source.history or [q.id for q in source.questions] != self.candidates:
            raise ValueError('native task candidate order or context changed')
        return tuple(self.compile_card(q, source.requirements) for q in source.questions)

    @torch.inference_mode()
    def score(self, model, device, provider, policy, rows, *, check_batch=True):
        records = {}
        def compiled():
            for asset, row in rows:
                uid = asset['id']
                if uid in records:
                    raise ValueError('duplicate recording in evaluation block')
                records[uid] = [None] * len(self.candidates)
                inputs = row['inputs']
                if not (inputs['input_ids'] == self.collate['audio_token_id']).any():
                    raise ValueError('cached recording misses its audio tokens')
                def bounded(items):
                    for item in items:
                        if item['inputs']['input_ids'].numel() > self.max_tokens:
                            raise ValueError('single-card context exceeds the frozen limit')
                        yield item
                yield bounded(render(inputs, self.tiles(uid), self.collate, uid, policy))

        def forward(group):
            logits = provider.forward(model, group, device=device, dtype=torch.bfloat16,
                                      policy_id=policy, **self.collate)
            if logits.shape != (len(group), 1):
                raise ValueError('each audio/card pair must have one logit')
            result = logits.sigmoid().float().flatten().cpu()
            if not torch.isfinite(result).all():
                raise ValueError('nonfinite pair probabilities')
            return result

        groups = (batches(compiled(), self.width, group_audio=False, collate=self.collate)
                  if self.max_chunks is None else
                  bounded_batches(compiled(), self.width, self.collate['audio_chunk_size'], self.max_chunks))
        for index, group in enumerate(groups):
            values = forward(group)
            if check_batch and index < self.parity_pairs and not torch.equal(values[:1], forward(group[:1])):
                raise ValueError('single-card batch differs from its singleton')
            for item, value in zip(group, values.tolist(), strict=True):
                uid, offset = item['request_id'].rsplit(':', 1)
                records[uid][int(offset)] = value
        if not records or any(None in row for row in records.values()):
            raise ValueError('incomplete audio/card predictions')
        return dict(policy_id=policy, records=[dict(id=uid, prob=row) for uid, row in records.items()])

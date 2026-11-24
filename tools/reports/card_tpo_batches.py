"""Read pinned audio features and forward one independent card per input."""
import torch

from multimodal_jev.context_audio_collate import forward_context_audio_batch
from multimodal_jev.ordered_prefetch import OrderedPrefetch
from multimodal_jev.readonly_encoded_audio_cache import ReadOnlyEncodedAudioCache
from tools.reports.card_scoring_benchmark import render, restore_binding


class CardTPOBatches:
    def __init__(self, batches, rows, cards, binding, cache_paths, prefetch, collate, stop, start=0):
        self.batches = batches
        self.rows = rows
        self.binding = binding
        self.cache = {False: ReadOnlyEncodedAudioCache(cache_paths['real']),
                      True: ReadOnlyEncodedAudioCache(cache_paths['synthetic'])}
        self.card_ids = cards['labels']
        selected = [cards['catalog'][i] for i in cards['arms'][binding['arm']]]
        if [c['id'] for c in selected] != self.card_ids:
            raise ValueError('frozen card order changed')
        self.templates = [tuple(torch.tensor(c[key], dtype=torch.long) for key in ('prefix', 'suffix'))
                          for c in selected]
        self.collate = collate
        self.cursor = start
        self.queue = OrderedPrefetch(self.read_batch, batches[start:stop], **prefetch)

    def read_batch(self, batch):
        audio = {}
        for slot in batch['slots']:
            uid = slot['id']
            if uid not in audio:
                row = self.rows[uid]
                binding = restore_binding(row, self.binding['acoustic_common'])
                audio[uid], _, _ = self.cache[row['synthetic']].read(binding)
        return batch['index'], audio

    def consume(self, step, policy):
        index, prepared = next(self.queue)
        if step != self.cursor or index != step:
            raise ValueError('card TPO batch cursor differs')
        self.cursor += 1
        items, metadata = [], []
        for i, slot in enumerate(self.batches[step]['slots']):
            uid = slot['id']
            if type(slot['padding']) is not bool or slot['weight'] != int(not slot['padding']):
                raise ValueError('card TPO padding mask differs')
            inp = prepared[uid]
            items.append(dict(request_id=f'{uid}::batch{step}-slot{i}', policy_id=policy,
                              inputs=inp, positions=None, request_weight=slot['weight'],
                              real_request=not slot['padding']))
            metadata.append(dict(id=uid))
        return items, metadata

    def forward(self, model, items, *, policy_id, **kwargs):
        if any(row['policy_id'] != policy_id for row in items):
            raise ValueError('card TPO forward policy differs')
        packed = [request for row in items for request in
                  render(row['inputs'], self.templates, self.collate, row['request_id'], policy_id)]
        values = forward_context_audio_batch(model, packed, policy_id=policy_id, **kwargs)
        if values.shape != (len(packed), 1):
            raise ValueError('single-card logits geometry differs')
        return values.reshape(len(items), len(self.templates))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.queue.close()


def check_inference(config):
    from pathlib import Path

    from multimodal_jev.baseline import file_hash
    from multimodal_jev.local_checkpoint import commit_json
    from tools.reports.build_tpo_view import read
    from tools.reports.evaluate_tpo_endpoints import start_model

    t = config['training']
    data, root, parent_path = (Path(t[k]) for k in ('targets', 'output', 'parent'))
    if file_hash(parent_path) != t['parent_sha'] or file_hash(data/'MANIFEST.json') != t['target_sha']:
        raise ValueError('inference parity source changed')
    parent, m = read(parent_path), read(data/'MANIFEST.json')
    for name in ('rows.json', 'batches.json', 'reference.pt', 'cards.json'):
        if file_hash(data/name) != m['files'][name]:
            raise ValueError('inference parity input changed')
    if file_hash(Path(parent['source_policy'])/'MANIFEST.json') != m['source_policy_sha256']:
        raise ValueError('inference parity checkpoint changed')
    torch.set_num_threads(parent['view_execution']['actor_threads'])
    device_id = t['physical_device']
    root.mkdir(parents=True, exist_ok=False)
    c = dict(model=parent['model'], base_id=parent['base_id'], contract=parent['source_contract'],
             snapshot=parent['source_policy'], policy_id=m['source_policy'], physical_devices=[device_id],
             max_model_len=parent['view_execution']['max_input_tokens'],
             process_device_scope=t['process_device_scope'])
    provider, model, device, _, collate, _ = start_model(c, root)
    rows = read(data/'rows.json')
    micro = m['schedule']['microbatch']
    start = t['probe_start']*micro
    slots = read(data/'batches.json')[0]['slots'][start:start+4*micro]
    probs = torch.load(data/'reference.pt', map_location='cpu', weights_only=True)
    reports = []
    with CardTPOBatches([dict(index=0, slots=slots)], rows, read(data/'cards.json'), m['card_input'],
                        parent['card_caches'], parent['view_execution']['prefetch'], collate, 1) as stream:
        items, metadata = stream.consume(0, m['source_policy'])
        with torch.inference_mode():
            for offset in range(0, len(items), micro):
                sample = items[offset:offset+micro]
                requests = [request for row in sample for request in
                            render(row['inputs'], stream.templates, collate, row['request_id'], m['source_policy'])]
                logits = provider.forward(model, requests, policy_id=m['source_policy'],
                                          device=device, dtype=torch.bfloat16, **collate)
                p = logits.sigmoid().float().reshape(len(sample), -1).cpu()
                q = probs[[rows[meta['id']]['index'] for meta in metadata[offset:offset+micro]]]
                error = float((p-q).abs().max())
                flips = int(((p>=.5)!=(q>=.5)).sum())
                reports.append(dict(rows=len(sample), max_abs=error, threshold_flips=flips,
                    passed=error<=parent['behavior_bounds']['probability_max_abs'] and flips==0))
    if len(reports) != 4 or not all(r['passed'] and r['rows'] == micro for r in reports):
        commit_json(root/'parity-failure.json', reports)
        raise ValueError('independent single-card probability parity failed')
    receipt = dict(passed=True, target_sha=t['target_sha'], parent_sha=t['parent_sha'],
                   source=config['source']['sha256'], physical_device=device_id, passes=reports)
    commit_json(root/'receipt.json', receipt)
    return receipt

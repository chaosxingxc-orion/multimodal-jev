"""SFC-Jev waveform-to-Omni padded actor inputs; no optimizer or model loading.

Each conversation turn is processed separately, with its ordered audio clips
together. Rebatching waveforms across unrelated requests can change STFT tail
frames, so batching below pads already computed features instead.
"""
import hashlib

from .baseline import file_hash
from .sfc_audio import load_sfc_audio
from .sfc_jev_runtime import messages, validate_bundle
from .sfc_reward_batch import compose, digest
from .omni_waveform_boundary import align_to_hop


def prepare(processor, requests, *, render=messages, waveform_boundary='raw'):
    """Input-only preparation, with file hashes checked before waveform loading.

    Caller pins processor assets and render implementation in its run snapshot.
    Outputs remain in CPU memory and must not be mutated between preparation
    and make_batch. No supervision is passed to the renderer or processor.
    waveform_boundary='vllm_hop' is an explicit versioned input change. Default
    raw preserves historical experiments; token evidence alone is not feature
    parity. Callers must register and verify the selected boundary separately.
    """
    import torch
    if waveform_boundary not in ('raw','vllm_hop'):
        raise ValueError('explicit raw or vllm_hop waveform boundary required')
    if not requests or len({r['id'] for r in requests}) != len(requests):
        raise ValueError('unique nonempty request roster required')
    prepared = {}
    audio_id = processor.tokenizer.convert_tokens_to_ids('<|audio_pad|>')
    for row in requests:
        validate_bundle(row)
        paths = row['input']['context']['audios']
        if len(paths) != len(row['audio_sha256']):
            raise ValueError('ordered audio hashes required')
        waves = []
        for path, pin in zip(paths, row['audio_sha256'], strict=True):
            if file_hash(path) != pin:
                raise ValueError('source audio changed')
            waves.append(load_sfc_audio(path)[0])
        original_samples=[len(w) for w in waves]
        if waveform_boundary=='vllm_hop':
            waves=align_to_hop(waves,processor.feature_extractor.hop_length)
        reference = None
        for decision in row['input']['decisions']:
            prompt = processor.apply_chat_template(render(row, decision['id'], waves),
                tokenize=False, add_generation_prompt=True)
            item = processor(text=[prompt], audio=waves, sampling_rate=16000,
                return_tensors='pt', padding=True, truncation=False)
            ids = item['input_ids']
            features = item['input_features']
            mask = item['feature_attention_mask']
            if (ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0
                    or not torch.all(item['attention_mask'] == 1)
                    or features.ndim != 3 or mask.ndim != 2
                    or features.shape[0] != len(waves)
                    or mask.shape != (len(waves), features.shape[-1])
                    or not torch.isfinite(features).all()
                    or not torch.all((mask == 0) | (mask == 1))):
                raise ValueError('untruncated unpadded prompt and finite aligned audio features required')
            hop = processor.feature_extractor.hop_length
            lengths = [len(w) for w in waves]
            expected = [min((n + hop - 1) // hop, max(lengths) // hop) for n in lengths]
            if mask.sum(-1).tolist() != expected:
                raise ValueError('waveform frame coverage differs')
            audio_positions = (ids[0] == audio_id).nonzero().flatten().tolist()
            spans = []
            for pos in audio_positions:
                if not spans or pos != spans[-1][-1] + 1:
                    spans.append([])
                spans[-1].append(pos)
            if len(spans) != len(waves):
                raise ValueError('ordered audio placeholder coverage differs')
            if reference is None:
                reference = (features, mask)
            else:
                if not torch.equal(features, reference[0]) or not torch.equal(mask, reference[1]):
                    raise ValueError('decision instruction changed audio features')
            # Sharing the verified feature tensors avoids copying them per decision.
            prepared[row['id'], decision['id']] = dict(
                request_sha256=digest(row), prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                input_ids_sha256=digest(ids[0].tolist()), input_ids=ids[0],
                input_features=reference[0], feature_attention_mask=reference[1],
                audio_token_lengths=list(map(len, spans)),
                waveform_boundary=waveform_boundary,original_samples=original_samples,
                processed_samples=[len(w) for w in waves])
    return prepared


def evidence(prepared):
    return {key: {name: value[name] for name in ('prompt_sha256', 'input_ids_sha256')}
            for key, value in prepared.items()}


def make_batch(processor, prepared, *, context_limit=4096, **reward_arguments):
    """Validate complete learning sampling groups, then construct verl DataProto.

    reward_arguments are the arguments to sfc_reward_batch.compose, except
    tokenizer and prompt_evidence, which are derived here from actual inputs.
    old_log_probs are sampled-token probabilities, never recomputed guesses.
    This entry point is specific to the padded, sequence-parallel-size-1 path.
    """
    import numpy as np
    import torch
    from verl import DataProto
    if type(context_limit) is not int or context_limit < 1:
        raise ValueError('positive context limit required')
    report = compose(**reward_arguments, tokenizer=processor.tokenizer,
                     prompt_evidence=evidence(prepared))
    records = reward_arguments['records']
    selected = [prepared[r['example_id'], r['decision_id']] for r in records]
    for record, item in zip(records, selected, strict=True):
        if (record['request_sha256'] != item['request_sha256']
                or digest(item['input_ids'].tolist()) != record['input_ids_sha256']):
            raise ValueError('prepared request/token identity changed')
    pmax = max(len(p['input_ids']) for p in selected)
    rmax = max(len(r['token_ids']) for r in records)
    if pmax + rmax > context_limit:
        raise ValueError('padded actor batch exceeds context limit; split microbatch without truncation')
    width = max(p['input_features'].shape[-1] for p in selected)
    pad = processor.tokenizer.pad_token_id
    if pad is None:
        raise ValueError('tokenizer padding token required')
    n = len(records)
    prompts = torch.full((n, pmax), pad, dtype=torch.long)
    responses = torch.full((n, rmax), pad, dtype=torch.long)
    attention = torch.zeros((n, pmax + rmax), dtype=torch.long)
    response_mask = torch.zeros((n, rmax), dtype=torch.long)
    old = torch.zeros((n, rmax), dtype=torch.float32)
    rewards = torch.zeros_like(old)
    multimodal = np.empty(n, dtype=object)
    source_groups = np.empty(n, dtype=object)
    identities = np.empty(n, dtype=object)
    groups = {r['id']: r['source_group'] for r in reward_arguments['requests']}
    cached_audio = {}
    for i, (record, item) in enumerate(zip(records, selected, strict=True)):
        plen, length = len(item['input_ids']), len(record['token_ids'])
        prompts[i, -plen:] = item['input_ids']
        responses[i, :length] = torch.tensor(record['token_ids'])
        attention[i, pmax - plen:pmax + length] = 1
        response_mask[i, :length] = 1
        old[i, :length] = torch.tensor(record['log_probs'])
        rewards[i, length - 1] = report['rewards'][i]
        key = record['example_id']
        if key not in cached_audio:
            extra = width - item['input_features'].shape[-1]
            cached_audio[key] = {name: torch.nn.functional.pad(item[name], (0, extra))
                                for name in ('input_features', 'feature_attention_mask')}
        multimodal[i] = cached_audio[key]
        source_groups[i] = groups[key]
        identities[i] = (key, record['decision_id'], record['replicate'])
    positions = attention.cumsum(-1) - 1
    positions.masked_fill_(attention == 0, 1)
    batch = DataProto.from_dict(tensors={
        'prompts': prompts, 'responses': responses,
        'input_ids': torch.cat((prompts, responses), dim=-1),
        'attention_mask': attention, 'position_ids': positions,
        'response_mask': response_mask, 'old_log_probs': old, 'token_level_rewards': rewards,
    }, non_tensors={
        'multi_modal_inputs': multimodal, 'uid': np.array(report['uid'], dtype=object),
        'source_group': source_groups, 'record_identity': identities,
    }, meta_info={
        'audio_contract': 'sfc-jev-qwen3-omni-padded-v1', 'group_size': report['group_size'],
        'requires_remove_padding': False, 'requires_sequence_parallel_size': 1,
        'context_limit': context_limit,
    })
    return batch, report

"""Label-free, full-roster sampled-response inputs for probability diagnostics.

Caller authenticates the sampling run, learning partition and processor assets.
This does not admit training data or replace the reward-aware training builder.
No synthetic responses, rewards, inferred EOS or replacement old probabilities.
"""
import math

from .sfc_reward_batch import digest, valid_hash


def build_components(processor, requests, prepared, records, *, policy_sha256,
                     instruction_sha256, group_size=2, context_limit=4096):
    import numpy as np
    import torch

    if (not valid_hash(policy_sha256) or not valid_hash(instruction_sha256)
            or type(group_size) is not int or group_size < 2
            or type(context_limit) is not int or context_limit < 1):
        raise ValueError('registered policy, instruction, group and context required')
    if (not requests or len({r['id'] for r in requests}) != len(requests)
            or any(r['split'] != 'train_id' for r in requests)):
        raise ValueError('unique learning TRAIN request roster required')
    roster = {(r['id'], d['id']): r for r in requests for d in r['input']['decisions']}
    if not roster or len(roster) != sum(len(r['input']['decisions']) for r in requests):
        raise ValueError('nonempty unique decision roster required')
    if set(prepared) != set(roster):
        raise ValueError('complete prepared decision roster required')
    expected = {(key[0], key[1], rep) for key in roster for rep in range(group_size)}
    actual = [(r['example_id'], r['decision_id'], r['replicate']) for r in records]
    if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('complete unique sampled replicate roster required')
    pad = processor.tokenizer.pad_token_id
    if type(pad) is not int or not 0 <= pad < len(processor.tokenizer):
        raise ValueError('valid tokenizer pad token required')
    selected = []
    for record in records:
        key = record['example_id'], record['decision_id']
        item = prepared[key]
        row = roster[key]
        ids, probs = record['token_ids'], record['log_probs']
        if (type(record['replicate']) is not int
                or record['policy_sha256'] != policy_sha256
                or record['instruction_sha256'] != instruction_sha256
                or record['request_sha256'] != item['request_sha256']
                or record['request_sha256'] != digest(row)
                or record['prompt_sha256'] != item['prompt_sha256']
                or not valid_hash(item['prompt_sha256'])
                or record['input_ids_sha256'] != item['input_ids_sha256']
                or record['input_ids_sha256'] != digest(item['input_ids'].tolist())
                or item['waveform_boundary'] != 'vllm_hop'):
            raise ValueError('sampled policy/prompt/audio boundary identity differs')
        if (type(ids) is not list or not 1 <= len(ids) <= 1024
                or any(type(t) is not int or not 0 <= t < len(processor.tokenizer) for t in ids)
                or type(probs) is not list or len(probs) != len(ids)
                or any(type(p) not in (int, float) or not math.isfinite(p) or p > 0 for p in probs)
                or record['finish_reason'] not in ('stop', 'length')
                or processor.tokenizer.decode(ids, skip_special_tokens=True) != record['raw_text']):
            raise ValueError('unaltered sampled tokens/text/logprobs required')
        prompt = item['input_ids']
        features, mask = item['input_features'], item['feature_attention_mask']
        if (prompt.ndim != 1 or prompt.numel() == 0 or prompt.dtype != torch.long
                or not torch.all((prompt >= 0) & (prompt < len(processor.tokenizer)))
                or features.ndim != 3 or mask.ndim != 2 or not features.is_floating_point()
                or features.shape[0] != len(row['input']['context']['audios'])
                or mask.shape != (features.shape[0], features.shape[-1])
                or features.device.type != 'cpu' or mask.device.type != 'cpu'
                or prompt.device.type != 'cpu' or not torch.isfinite(features).all()
                or not torch.all((mask == 0) | (mask == 1))
                or not torch.all(mask.sum(-1) > 0)
                or not torch.equal(mask, mask.cumprod(-1))):
            raise ValueError('complete finite CPU prompt and ordered audio features required')
        selected.append(item)
    pmax = max(len(v['input_ids']) for v in selected)
    rmax = max(len(r['token_ids']) for r in records)
    if pmax + rmax > context_limit:
        raise ValueError('full padded context overflow; no truncation permitted')
    width = max(v['input_features'].shape[-1] for v in selected)
    n = len(records)
    prompts = torch.full((n, pmax), pad, dtype=torch.long)
    responses = torch.full((n, rmax), pad, dtype=torch.long)
    attention = torch.zeros((n, pmax+rmax), dtype=torch.long)
    response_mask = torch.zeros((n, rmax), dtype=torch.long)
    old = torch.zeros((n, rmax), dtype=torch.float32)
    audio = np.empty(n, dtype=object)
    identities = np.empty(n, dtype=object)
    for i, (r, item) in enumerate(zip(records, selected, strict=True)):
        plen, length = len(item['input_ids']), len(r['token_ids'])
        prompts[i, -plen:] = item['input_ids']
        responses[i, :length] = torch.tensor(r['token_ids'])
        attention[i, pmax-plen:pmax+length] = 1
        response_mask[i, :length] = 1
        old[i, :length] = torch.tensor(r['log_probs'], dtype=torch.float32)
        audio[i] = {k: torch.nn.functional.pad(item[k], (0, width-item[k].shape[-1]))
                    for k in ('input_features', 'feature_attention_mask')}
        identities[i] = actual[i]
    if not torch.isfinite(old[response_mask.bool()]).all():
        raise ValueError('sampled logprobs must remain finite in actor FP32 storage')
    positions = attention.cumsum(-1)-1
    positions.masked_fill_(attention == 0, 1)
    return dict(tensors=dict(prompts=prompts, responses=responses,
        input_ids=torch.cat((prompts, responses), dim=-1), attention_mask=attention,
        position_ids=positions, response_mask=response_mask, old_log_probs=old),
        non_tensors=dict(multi_modal_inputs=audio, record_identity=identities),
        meta_info=dict(audio_contract='sfc-jev-qwen3-omni-padded-v1',
            requires_remove_padding=False, requires_sequence_parallel_size=1,
            diagnostic_only=True, targets_read=False, synthetic_old_log_probs=False,
            waveform_boundary='vllm_hop', group_size=group_size,
            policy_sha256=policy_sha256, instruction_sha256=instruction_sha256,
            records_sha256=digest(records), context_limit=context_limit))


def make_probability_batch(processor, requests, prepared, records, **kwargs):
    from verl import DataProto

    return DataProto.from_dict(**build_components(processor, requests, prepared, records, **kwargs))

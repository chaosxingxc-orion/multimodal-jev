"""Bind padded actor probabilities to complete, original sampled responses.

Use only with the label-free probability builder and compute_log_probs adapter.
Runtime/model/features authentication remains the runner's responsibility.
"""
from .omni_probability_audit import _key, _probabilities
from .sfc_reward_batch import digest


def _aligned(data, records):
    import torch

    meta = data.meta_info
    if (meta.get('audio_contract') != 'sfc-jev-qwen3-omni-padded-v1'
            or meta.get('diagnostic_only') is not True
            or meta.get('targets_read') is not False
            or meta.get('synthetic_old_log_probs') is not False
            or meta.get('waveform_boundary') != 'vllm_hop'
            or meta.get('records_sha256') != digest(records)):
        raise ValueError('authenticated complete probability diagnostic batch required')
    original = {}
    for record in records:
        key = _key(record)
        _probabilities(record['token_ids'], record['log_probs'])
        if (key in original or record['policy_sha256'] != meta.get('policy_sha256')
                or record['instruction_sha256'] != meta.get('instruction_sha256')
                or record['finish_reason'] not in ('stop', 'length')):
            raise ValueError('unique original policy/instruction responses required')
        original[key] = record
    tensors = data.batch
    responses, mask, old = (tensors[k] for k in ('responses', 'response_mask', 'old_log_probs'))
    identities = data.non_tensor_batch['record_identity']
    if (not original or responses.ndim != 2 or responses.dtype != torch.long
            or mask.shape != responses.shape or old.shape != responses.shape
            or old.dtype != torch.float32 or len(identities) != len(original)
            or len(responses) != len(original)
            or not torch.all((mask == 0) | (mask == 1))
            or not torch.equal(mask, mask.cumprod(-1))
            or not torch.isfinite(old[mask.bool()]).all()):
        raise ValueError('complete right-padded response tensor roster required')
    rows = []; seen = set()
    for i, identity in enumerate(identities):
        if (not isinstance(identity, (list, tuple)) or len(identity) != 3
                or type(identity[0]) is not str or type(identity[1]) is not str
                or type(identity[2]) is not int):
            raise ValueError('exact response identity tuple required')
        key = tuple(identity)
        if key not in original or key in seen:
            raise ValueError('unknown or duplicate response identity')
        seen.add(key)
        record = original[key]
        length = len(record['token_ids'])
        if (int(mask[i].sum().item()) != length
                or responses[i, :length].detach().cpu().tolist() != record['token_ids']
                or not torch.equal(old[i, :length].detach().cpu(),
                                   torch.tensor(record['log_probs'], dtype=torch.float32))):
            raise ValueError('sampled tokens, mask or old probabilities changed')
        rows.append(record)
    return rows


def capture(data, log_probs, records, *, name):
    """Export valid tokens, retaining actor dtype precision and original identity.

    A full DataProto reorder is accepted; a partial batch is rejected. Padding
    logprobs are ignored (including NaN); valid-token values are never repaired.
    This cannot detect a caller passing probabilities from a different forward.
    """
    import torch

    if type(name) is not str or not name:
        raise ValueError('registered run name required')
    rows = _aligned(data, records)
    if (not isinstance(log_probs, torch.Tensor) or log_probs.ndim != 2
            or not log_probs.is_floating_point()
            or log_probs.shape != data.batch['responses'].shape):
        raise ValueError('aligned floating actor response probabilities required')
    values = log_probs.detach().cpu()
    exported = []
    for i, record in enumerate(rows):
        tokens = list(record['token_ids'])
        probs = values[i, :len(tokens)].tolist()
        _probabilities(tokens, probs)
        exported.append(dict(example_id=record['example_id'], decision_id=record['decision_id'],
                             replicate=record['replicate'], record_sha256=digest(record),
                             token_ids=tokens, log_probs=probs))
    return dict(name=name, responses=exported)


def recompute(actor, data, records, *, name, micro_batch_size, pad_token_id):
    """One forward, then capture in the same row order; no retry or resampling."""
    import torch
    from .omni_actor_forward import compute_log_probs

    if type(name) is not str or not name:
        raise ValueError('registered run name required')
    _aligned(data, records)  # Reject broken identity before spending a forward.
    with torch.no_grad():
        log_probs = compute_log_probs(actor, data, micro_batch_size=micro_batch_size,
                                      pad_token_id=pad_token_id)
    return capture(data, log_probs, records, name=name)


def recompute_native_observed(actor, data, records, *, name, micro_batch_size, pad_token_id):
    """Observe one native NPU recomputation and bind its static microbatch order.

    Diagnostic only; no numerical threshold or retry. Runtime/model/rank identity
    authentication remains the runner's responsibility, as for ``recompute``.
    """
    import torch
    from .omni_actor_forward import compute_log_probs
    from .omni_logprob_observer import observe_native_logprobs, bind_observations

    if type(name) is not str or not name:
        raise ValueError('registered run name required')
    _aligned(data, records)
    with torch.no_grad(), observe_native_logprobs() as report:
        log_probs = compute_log_probs(actor, data, micro_batch_size=micro_batch_size,
                                      pad_token_id=pad_token_id)
    return bind_observations(data, log_probs, records, report,
                             name=name, micro_batch_size=micro_batch_size)

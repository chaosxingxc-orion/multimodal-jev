"""Explicit batch encode for a separately constructed segmented context owner.

Wire identities are identical to FixedContextRollout._score. This module does
not allocate models, change policies or provide its own inference fallback.
"""
import hashlib
import json

import torch

from multimodal_jev.span_request_identity import tensor_identity


def prepare_request(owner, row, audio_token_id, params_class):
    if (type(row) is not dict or set(row) != {'inputs', 'positions'}
            or type(row['inputs']) is not dict
            or set(row['inputs']) != {'input_ids', 'attention_mask', 'input_features', 'input_features_mask'}
            or any(not isinstance(t, torch.Tensor) or t.device.type != 'cpu'
                   for t in [*row['inputs'].values(), row['positions']])):
        raise ValueError('label-free complete CPU request required')
    inputs, positions = row['inputs'], row['positions']
    ids, mask = inputs['input_ids'], inputs['attention_mask']
    if (ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype != torch.long
            or mask.shape != ids.shape or not (mask == 1).all() or (ids < 0).any()):
        raise ValueError('complete unpadded integer prompt required')
    ids = ids[0].tolist()
    marker = owner.bundle['contract']['token']['id']
    if (type(audio_token_id) is not int or audio_token_id < 0 or audio_token_id == marker
            or len(ids) < 2 or len(ids) > owner.max_model_len
            or positions.dtype != torch.long or positions.ndim != 2 or positions.shape[0] != 1
            or not 0 < positions.shape[1] <= owner.max_decisions
            or (marker == 151706 and 151705 in ids)):
        raise ValueError('bounded complete context and distinct markers required')
    markers = [i for i, token in enumerate(ids) if token == marker]
    at = [i for i, token in enumerate(ids) if token == audio_token_id]
    if positions[0].tolist() != markers or not at or at != list(range(at[0], at[-1]+1)):
        raise ValueError('exact marker coverage and contiguous audio required')
    features, feature_mask = inputs['input_features'], inputs['input_features_mask']
    if (features.ndim != 3 or features.shape[0] != 1 or not features.is_floating_point()
            or min(features.shape) < 1 or not torch.isfinite(features).all()
            or feature_mask.shape != (1, features.shape[-1])
            or not ((feature_mask == 0) | (feature_mask == 1)).all()):
        raise ValueError('finite audio features and binary mask required')
    frames = int(feature_mask.sum())
    if frames < 1 or not feature_mask[0, :frames].bool().all() or feature_mask[0, frames:].bool().any():
        raise ValueError('contiguous complete audio frames required')
    identity = dict(policy_id=owner.bundle['policy_id'], positions=markers,
                    inputs={k: tensor_identity(v) for k, v in sorted(inputs.items())})
    rid = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    prompt = dict(prompt_token_ids=ids[:at[0]]+[audio_token_id]+ids[at[-1]+1:],
        multi_modal_data={'audio': dict(input_audio_features=features[0, :, :frames].contiguous(),
            audio_feature_lengths=torch.tensor([frames]), feature_attention_mask=feature_mask[:, :frames].contiguous())},
        multi_modal_uuids={'audio': [rid]})
    params = params_class(use_activation=False, extra_kwargs=dict(policy_id=identity['policy_id'],
        marker_positions=markers, expected_input_ids=ids))
    return dict(prompt=prompt, params=params, ids=ids, request_id=rid, request_identity=identity)


def encode_many(owner, rows, *, audio_token_id, params_class):
    """Caller holds the same lock as policy switching for the whole operation."""
    if type(rows) is not list or not 1 <= len(rows) <= owner.max_batch_queries:
        raise ValueError('nonempty bounded batch required')
    prepared = [prepare_request(owner, row, audio_token_id, params_class) for row in rows]
    if sum(len(p['ids']) for p in prepared) > owner.max_batch_tokens:
        raise ValueError('expanded batch token capacity exceeded')
    if len({p['request_id'] for p in prepared}) != len(prepared):
        raise ValueError('duplicate request identities must be resolved before batching')
    policy_id = owner.bundle['policy_id']
    results = owner.engine.encode([p['prompt'] for p in prepared],
        pooling_params=[p['params'] for p in prepared], pooling_task='token_classify',
        use_tqdm=False, lora_request=owner.lora)
    if len(results) != len(prepared) or owner.bundle['policy_id'] != policy_id:
        raise ValueError('complete results under one unchanged policy required')
    records = []
    for p, result in zip(prepared, results, strict=True):
        prob = result.outputs.data
        if (result.prompt_token_ids != p['ids'] or not isinstance(prob, torch.Tensor)
                or prob.shape != (len(p['request_identity']['positions']), 1)
                or not prob.is_floating_point() or not torch.isfinite(prob).all()
                or ((prob < 0) | (prob > 1)).any()):
            raise ValueError('ordered complete prompts and finite probability outputs required')
        records.append(dict(request_id=p['request_id'], request_identity=p['request_identity'],
                            prob=prob[:, 0].float().tolist(), policy_id=policy_id))
    return records

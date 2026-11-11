"""Bind exact actor/rollout evidence independently of source grouping or gold."""
import hashlib
import json


def tensor_identity(tensor):
    import torch
    value=tensor.detach().cpu().contiguous()
    return dict(shape=list(value.shape),dtype=str(value.dtype),
                sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())


def bind_span_request(request, audio_receipt, encoded_inputs, positions, candidate_ids,
                      policy_id, temperature=1.0):
    """Request ID includes fixed augmentation and policy, never answers/rewards.

    Candidate IDs are the precise ordered token support per Span. The actor and
    rollout must bind this common pre-backend representation before GPU/NPU
    casts. Worker-expanded prompt equality remains a separate execution check.
    """
    import torch
    if not isinstance(policy_id,str) or not policy_id or temperature!=1.0:
        raise ValueError('pinned policy and untransformed candidate sampling required')
    required={'input_ids','attention_mask','input_features','input_features_mask'}
    if not required <= encoded_inputs.keys() or set(encoded_inputs)-required-{'position_ids'}:
        raise ValueError('explicit known label-free audio inputs required')
    if (positions.dtype!=torch.long or positions.shape!=(1,len(request.questions)) or
            encoded_inputs['input_ids'].shape[0]!=1 or len(candidate_ids)!=len(request.questions)):
        raise ValueError('one complete aligned Span request required')
    for ids in candidate_ids:
        if len(ids)<2 or len(set(ids))!=len(ids) or any(type(i)!=int or i<0 for i in ids):
            raise ValueError('distinct ordered candidate token support required')
    for key in ('input_waveform_sha256','output_waveform_sha256'):
        value=audio_receipt.get(key)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('processed waveform identity required')
    if 'config' not in audio_receipt or audio_receipt.get('sample_rate')!=16000:
        raise ValueError('processing configuration and sample rate required')
    payload=dict(version='jev-span-request-identity-v1',request=request.model_input(),
                 audio_processing=audio_receipt,inputs={k:tensor_identity(v) for k,v in sorted(encoded_inputs.items())},
                 positions=positions.tolist(),candidate_ids=candidate_ids,policy_id=policy_id,
                 sampling=dict(temperature=temperature,policy='independent_finite_categorical'))
    serialized=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(serialized).hexdigest(),payload

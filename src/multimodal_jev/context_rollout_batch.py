"""Bind one authenticated vLLM probability vector to a complete sampled group."""
import hashlib
import json
import torch

from .span_request_identity import tensor_identity


def bind_rollout(record,inputs,positions,*,policy_id,group_size,seed):
    if (type(group_size) is not int or group_size<2 or type(seed) is not int or seed<0 or
            not isinstance(policy_id,str) or not policy_id):
        raise ValueError('explicit policy and sampling configuration required')
    identity=dict(policy_id=policy_id,positions=positions[0].tolist(),
        inputs={k:tensor_identity(v) for k,v in sorted(inputs.items())})
    rid=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if (record['policy_id']!=policy_id or record['request_identity']!=identity or record['request_id']!=rid):
        raise ValueError('rollout policy or actual audio input differs')
    prob=torch.tensor(record['prob'],dtype=torch.float32)
    if (prob.shape!=(positions.shape[1],) or not torch.isfinite(prob).all() or
            ((prob<=0)|(prob>=1)).any()):
        # Saturated probability output loses finite log odds. Never clamp and
        # misrepresent a different distribution as the engine behavior policy.
        raise ValueError('strictly interior finite probability vector required')
    logp=torch.stack((torch.log1p(-prob),torch.log(prob)),-1)
    behavior=tuple(v[None] for v in logp.unbind(0))
    generator=torch.Generator(device='cpu').manual_seed(seed)
    actions=(torch.rand((1,group_size,len(prob)),generator=generator)<prob).long()
    return dict(behavior=behavior,actions=actions,receipt=dict(engine_request_id=rid,policy_id=policy_id,
        group_size=group_size,seed=seed,probability=tensor_identity(prob),actions=tensor_identity(actions),
        logprob_source='FP32 log1p(-p)/log(p) from actual vLLM sigmoid output; actor must verify rounding gap'))

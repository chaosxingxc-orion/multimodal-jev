"""Independent CPU replay of recorded online behavior and reference outputs."""
import hashlib
import json

import torch

from multimodal_jev.span_request_identity import tensor_identity


def verify_sample(entry, group, *, update, policy_id, reference_policy_id):
    """Check saved raw outputs, FP32 logs and deterministic Bernoulli sampling.

    This verifies recorded evidence, not that the NPU produced the records.
    Source/container identities and actor/engine parity are separate checks.
    """
    def require(condition, message):
        if not condition:raise ValueError(message)

    require(entry['request_id']==group['request_id'] and entry['update']==update,
            'scheduled request differs')
    receipt=entry['rollout'];actions=group['actions']
    require(group['policy_id']==receipt['policy_id']==policy_id,'behavior policy differs')
    seed=int.from_bytes(hashlib.sha256(repr((317,update,group['request_id'])).encode()).digest()[:8],'big') % (2**63)
    require(receipt['seed']==seed and receipt['group_size']==16,'sampling configuration differs')
    require(actions.shape==(1,16,60) and actions.dtype==torch.int64,'complete action group required')
    require(receipt['actions']==tensor_identity(actions),'sample hash differs')
    for kind,policy,logs in [('behavior',policy_id,group['behavior']),
                             ('reference',reference_policy_id,group['reference_policy'])]:
        record=entry[kind+'_record'];identity=record['request_identity']
        require(record['policy_id']==identity['policy_id']==policy,'record policy differs')
        rid=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        require(rid==record['request_id'],'input identity hash differs')
        probability=torch.tensor(record['prob'],dtype=torch.float32)
        require(probability.shape==(60,) and torch.isfinite(probability).all() and
                ((probability>0)&(probability<1)).all(),'finite interior probabilities required')
        expected=torch.stack((torch.log1p(-probability),torch.log(probability)),-1)
        require(len(logs)==60 and all(x.shape==(1,2) for x in logs) and
                torch.equal(torch.cat(logs,dim=0),expected),'saved log probabilities differ')
        if kind=='behavior':
            require(rid==receipt['engine_request_id'] and tensor_identity(probability)==receipt['probability'],
                    'behavior record hash differs')
            draw=torch.rand((1,16,60),generator=torch.Generator().manual_seed(seed))
            require(torch.equal((draw<probability).long(),actions),'sample replay differs')
        else:
            require(entry['reference_policy_id']==reference_policy_id and
                    rid==entry['reference_request_id'],'fixed reference differs')
            original=entry['behavior_record']['request_identity']
            require(identity['inputs']==original['inputs'] and identity['positions']==original['positions'],
                    'reference and behavior inputs differ')
    return dict(sample_replay_exact=True,behavior_logs_exact=True,reference_logs_exact=True)

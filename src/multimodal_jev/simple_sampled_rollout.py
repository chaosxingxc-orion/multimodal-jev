"""Label-free G16 sampling from authenticated, version-bound vLLM outputs.

The caller verifies execution receipts and HF/vLLM parity before this bridge.
This binds transport identity and samples only; it grants no NPU admission.
"""
import hashlib
import json
import math
import torch
from .span_policy import sample_spans


def sample_record(record, *, expected_request_id, expected_identity,
                  policy_id, seed, update, slot, group_size=16):
    if (type(seed) is not int or not 0 <= seed < 2**63
            or type(update) is not int or update < 0
            or type(slot) is not int or slot < 0 or group_size != 16
            or type(group_size) is not int or not isinstance(policy_id,str) or not policy_id):
        raise ValueError('fixed G16 and explicit valid sampling coordinates required')
    identity = record['request_identity']
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()
    if (record['request_id'] != digest or digest != expected_request_id
            or identity != expected_identity or identity.get('policy_id') != policy_id
            or identity.get('sampling') != dict(temperature=1.0,policy='independent_finite_categorical')
            or len(identity.get('candidate_ids', [])) != 1
            or len(identity['candidate_ids'][0]) != 60
            or len(set(identity['candidate_ids'][0])) != 60):
        raise ValueError('exact policy/input/native60 candidate identity required')
    # No labels, target-dependent group filtering, rejection sampling, or argmax.
    raw = record['candidate_logprobs']
    if len(raw) != 1 or len(raw[0]) != 60:
        raise ValueError('complete single Final60 behavior distribution required')
    behavior = (torch.tensor(raw[0],dtype=torch.float32).reshape(1,60),)
    coordinate = dict(schema='simple-final-sampling-v1',seed=seed,update=update,slot=slot,
                      request_id=digest,policy_id=policy_id,group_size=group_size)
    key = hashlib.sha256(json.dumps(coordinate,sort_keys=True,separators=(',', ':')).encode()).hexdigest()
    local_seed = int(key[:16],16) % 2**63
    actions, old_joint = sample_spans(behavior,group_size,torch.Generator(device='cpu').manual_seed(local_seed))
    return dict(behavior=behavior,actions=actions,old_joint=old_joint,
                receipt=dict(**coordinate,sampling_sha256=key,torch_seed=local_seed,
                             actions=actions.tolist(),old_joint=old_joint.tolist(),
                             torch_version=torch.__version__,replacement=True))


def fixed_reference(record, *, expected_identity, reference_policy_id):
    """Require identical input/support with only the immutable policy changed."""
    expected = json.loads(json.dumps(expected_identity))
    expected['policy_id'] = reference_policy_id
    identity = record['request_identity']
    digest = hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()
    if identity != expected or record['request_id'] != digest:
        raise ValueError('fixed reference policy must use identical audio/task/support')
    raw = record['candidate_logprobs']
    if len(raw) != 1 or len(raw[0]) != 60:
        raise ValueError('complete fixed reference native60 required')
    result = (torch.tensor(raw[0],dtype=torch.float32).reshape(1,60),)
    if (not torch.isfinite(result[0]).all()
            or not torch.allclose(result[0].logsumexp(-1),torch.zeros(1),atol=1e-6,rtol=0)):
        raise ValueError('finite normalized fixed reference required')
    return result


def actor_item(item, behavior_record, reference_record, *, expected_identity,
               reference_policy_id, seed, update, slot):
    """Bind prepared actor tensors before attaching external target metadata.

    `item` must come from the authenticated learning schedule/target compiler.
    This function cannot certify the dataset release or backend execution.
    """
    from .span_request_identity import tensor_identity
    required = {'input_ids','attention_mask','input_features','input_features_mask'}
    inputs = item['inputs']
    if (not required <= inputs.keys() or set(inputs)-required-{'position_ids'}
            or any(v.device.type != 'cpu' for v in inputs.values())
            or expected_identity.get('inputs') != {k:tensor_identity(v) for k,v in sorted(inputs.items())}
            or expected_identity.get('positions') != item['positions'].tolist()
            or expected_identity.get('candidate_ids') != item['candidates']
            or expected_identity.get('policy_id') != item['policy_id']):
        raise ValueError('actor tensors/support must match exact behavior input')
    y = item['targets']; mask = item['supervised']; weights = item['span_weights']
    if (y.dtype != torch.long or y.shape != (1,1) or y.device.type != 'cpu'
            or not 0 <= int(y[0,0]) < 60 or mask.dtype != torch.bool or mask.shape != (1,1)
            or not bool(mask.all()) or weights.shape != (1,1) or not bool((weights == 1).all())):
        raise ValueError('one fully supervised native Final60 target required')
    digest = hashlib.sha256(json.dumps(expected_identity,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()
    sampled = sample_record(behavior_record,expected_request_id=digest,expected_identity=expected_identity,
                            policy_id=item['policy_id'],seed=seed,update=update,slot=slot)
    reference = fixed_reference(reference_record,expected_identity=expected_identity,
                                reference_policy_id=reference_policy_id)
    # Slot identity preserves duplicates used as zero-weight schedule padding.
    uid = f'{digest}::update{update}-slot{slot}'
    packed = dict(item,request_id=uid,behavior=sampled['behavior'],actions=sampled['actions'],
                  reference_policy=reference,reference_policy_id=reference_policy_id)
    return packed,dict(sampled['receipt'],actor_request_id=uid,reference_policy_id=reference_policy_id,
                       reference_request_id=reference_record['request_id'])


def actor_batch(prepared, behavior_result, reference_result, *, policy_id,
                reference_policy_id, seed, update):
    """Join an entire prepared global batch before splitting across DP ranks.

    Each prepared entry holds its compiled identity and external actor metadata.
    Backend records cover unique exact requests; repeated zero-weight padding
    slots still get distinct actor identities and G16 draws. No rows are dropped.
    """
    if not prepared or len(prepared) % 4:
        raise ValueError('complete equal fourrank global batch required')
    def index(result, policy):
        if (result.get('passed') is not True or result.get('backend') != 'vllm'
                or result.get('base_id','')+':'+result.get('adapter_id','') != policy):
            raise ValueError('successful exact vLLM policy result required')
        rows = result['records']; indexed = {r['request_id']:r for r in rows}
        if len(rows) != len(indexed):
            raise ValueError('duplicate backend request identity')
        return indexed
    behavior = index(behavior_result,policy_id)
    reference = index(reference_result,reference_policy_id)
    items=[];receipts=[];expected_behavior=set();expected_reference=set()
    for slot,entry in enumerate(prepared):
        identity=entry['identity'];item=entry['item']
        if identity.get('policy_id') != policy_id or item['policy_id'] != policy_id:
            raise ValueError('entire prepared batch must use current behavior version')
        if (type(item.get('real_request')) is not bool
                or type(item.get('request_weight')) not in (float,int)
                or not math.isfinite(item['request_weight'])
                or item['request_weight']<0
                or (not item['real_request'] and item['request_weight']!=0)):
            raise ValueError('explicit real requests and zero padding weights required')
        rid=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()
        ref_identity=dict(identity,policy_id=reference_policy_id)
        refid=hashlib.sha256(json.dumps(ref_identity,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()
        if rid not in behavior or refid not in reference:
            raise ValueError('missing behavior or fixed reference request')
        packed,receipt=actor_item(item,behavior[rid],reference[refid],expected_identity=identity,
            reference_policy_id=reference_policy_id,seed=seed,update=update,slot=slot)
        items.append(packed);receipts.append(receipt);expected_behavior.add(rid);expected_reference.add(refid)
    if behavior.keys()!=expected_behavior or reference.keys()!=expected_reference:
        raise ValueError('backend population must exactly cover the global batch')
    real=sum(item['real_request'] for item in items)
    if real<1 or sum(item['request_weight'] for item in items)<=0:
        raise ValueError('positive real learning population required')
    return items,dict(policy_id=policy_id,reference_policy_id=reference_policy_id,update=update,
                      global_rows=len(items),global_real_requests=real,group_size=16,
                      unique_behavior_requests=len(behavior),sampling=receipts,
                      scope='Complete version/input-bound sampling transport; caller authenticates runtime and learning release.')

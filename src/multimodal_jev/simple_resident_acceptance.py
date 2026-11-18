"""Post-update all-candidate parity with exact global-slot coverage, including padding."""
import hashlib
import json
from .span_parity import compare_span_policies


def compare_updated_batch(resident_ranks, rollout, sampling, *, policy_id, updates):
    if (type(updates) is not int or updates<1 or sampling['update']!=updates-1
            or sampling['global_rows']!=128 or len(sampling['sampling'])!=128
            or len(resident_ranks)!=4 or any(len(r)!=32 for r in resident_ranks)
            or rollout.get('passed') is not True or rollout.get('backend')!='vllm'
            or rollout.get('base_id','')+':'+rollout.get('adapter_id','')!=policy_id
            or rollout.get('reference_compared') is not False):
        raise ValueError('complete post-update DP4/native128 and matching rollout required')
    old_to_new={}
    for row in rollout['records']:
        identity=row['request_identity']
        wire=json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False)
        if identity['policy_id']!=policy_id or hashlib.sha256(wire.encode()).hexdigest()!=row['request_id']:
            raise ValueError('updated policy/input identity differs')
        old=dict(identity,policy_id=sampling['policy_id'])
        old_id=hashlib.sha256(json.dumps(old,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if old_id in old_to_new:raise ValueError('duplicate updated backend input')
        old_to_new[old_id]=row
    resident=[row for rank in resident_ranks for row in rank]
    by_id={row['request_id']:row for row in resident}
    if len(by_id)!=128:raise ValueError('duplicate global actor slot')
    checks=[];used=set()
    for slot,record in enumerate(sampling['sampling']):
        uid=f'{record["request_id"]}::update{updates-1}-slot{slot}'
        if (record['policy_id']!=sampling['policy_id'] or record['update']!=updates-1
                or record['slot']!=slot or record['actor_request_id']!=uid
                or uid not in by_id or record['request_id'] not in old_to_new):
            raise ValueError('scheduled actor slot or behavior lineage differs')
        checks.append(compare_span_policies(by_id[uid]['candidate_logprobs'],
            old_to_new[record['request_id']]['candidate_logprobs']))
        used.add(record['request_id'])
    if used!=old_to_new.keys() or not all(c['passed'] for c in checks):
        raise ValueError('full updated batch coverage/probability parity failed')
    return dict(passed=True,policy_id=policy_id,updates=updates,requests=128,
        unique_inputs=len(used),maximum_logprob_error=max(c['maximum_logprob_error'] for c in checks),
        argmax_changes=sum(len(c['argmax_changed_spans']) for c in checks))

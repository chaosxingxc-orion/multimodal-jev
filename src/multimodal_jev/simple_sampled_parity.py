"""Bind post-update resident outputs to fresh backend requests and compare all60."""
import hashlib
import json
from .span_parity import compare_span_policies


def compare_resident(resident, backend, sampling, *, old_policy_id, new_policy_id, update=0):
    if type(update) is not int or update not in (0,1):
        raise ValueError('explicit first or second engineering update required')
    if len(resident)!=128 or len(sampling)!=128 or len(backend)!=256:
        raise ValueError('complete resident128 and fresh backend256 required')
    by_id={r['request_id']:r for r in resident}
    if len(by_id)!=128:raise ValueError('duplicate resident output')
    # Fresh HF/vLLM covers the original two-batch prefix. Only first128 was
    # consumed by the engineering RL update; compare exact identities.
    old_to_new={}
    for row in backend:
        identity=row['request_identity']
        wire=json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False)
        if (identity['policy_id']!=new_policy_id
                or hashlib.sha256(wire.encode()).hexdigest()!=row['request_id']):
            raise ValueError('fresh backend policy/request identity differs')
        prior=dict(identity,policy_id=old_policy_id)
        prior_id=hashlib.sha256(json.dumps(prior,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if prior_id in old_to_new:raise ValueError('duplicate backend input')
        old_to_new[prior_id]=row
    checks=[];used=set()
    for slot,record in enumerate(sampling):
        rid=record['request_id'];uid=f'{rid}::update{update}-slot{slot}'
        if (record['policy_id']!=old_policy_id or record['update']!=update or record['slot']!=slot
                or record['actor_request_id']!=uid or uid not in by_id or rid not in old_to_new):
            raise ValueError('complete original scheduled sampling identity required')
        a=by_id[uid]['candidate_logprobs'];b=old_to_new[rid]['candidate_logprobs']
        if len(a)!=1 or len(b)!=1 or len(a[0])!=60 or len(b[0])!=60:
            raise ValueError('complete Final60 required')
        check=compare_span_policies(a,b)
        if not check['passed']:raise ValueError('resident/fresh backend probability gate failed')
        used.add(uid);checks.append(check)
    if used!=set(by_id):raise ValueError('resident coverage differs')
    return dict(passed=True,requests=128,maximum_logprob_error=max(c['maximum_logprob_error'] for c in checks),
        argmax_changed_spans=sum(len(c['argmax_changed_spans']) for c in checks))

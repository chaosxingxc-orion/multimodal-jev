"""Authenticate augmented learning rows and select whole source-view groups.

No training dispatch or source-policy approval is implied by successful loading.
The actor must encode and bind each selected input to accepted behavior records.
"""
import json
import hashlib
from pathlib import Path
from .baseline import file_hash
from .slurp_intent_pilot import load_pilot
from .intent_binding_training import expand
from .intent_actor_inputs import schedule
from .span_candidate_intervention import from_input,VARIANTS
from .slurp_training import content_hash


def load(root, manifest_sha256, arm):
    root=Path(root)
    if arm not in ('D','Q','P') or file_hash(root/'MANIFEST.json')!=manifest_sha256:
        raise ValueError('fixed augmented manifest and arm required')
    manifest=json.loads((root/'MANIFEST.json').read_text())
    if (manifest['schema']!='slurp-intent-binding-training-candidate-v1' or manifest['role']!='learning'
            or manifest['variants']!=list(VARIANTS) or manifest['status']!='candidate_only_not_actor_admitted'):
        raise ValueError('registered learning candidate required')
    expected={a+'-'+k+'.json' for a in ('D','Q','P') for k in ('requests','targets')}
    if set(manifest['files'])!=expected:raise ValueError('complete fixed arm files required')
    for name,h in manifest['files'].items():
        if file_hash(root/name)!=h:raise ValueError('augmented release changed')
    for path,h in manifest['implementation_sha256'].items():
        if file_hash(path)!=h:raise ValueError('registered compiler changed')
    if (root/'Q-requests.json').read_bytes()!=(root/'P-requests.json').read_bytes():
        raise ValueError('Q/P inputs differ')
    sources,admission=load_pilot(manifest['source_pilot'],manifest['source_manifest_sha256'],arm)
    requests=json.loads((root/f'{arm}-requests.json').read_text())
    targets=json.loads((root/f'{arm}-targets.json').read_text())
    expected_rows=[pair for source in sources for pair in expand(source)]
    if len(requests)!=len(expected_rows) or len(targets)!=len(expected_rows):
        raise ValueError('complete source/view coverage required')
    rows=[];wire='D' if arm=='D' else 'Q'
    for row,target,(er,et) in zip(requests,targets,expected_rows,strict=True):
        if content_hash(row)!=content_hash(er) or target!=et:
            raise ValueError('augmented input or supervision differs from reviewed source')
        rows.append(dict(id=row['id'],source_id=row['source_id'],variant=row['variant'],
            request=from_input(row['request']),target=target,audio_sha256=row['audio_sha256'],
            gate_id=row['source_id']+'::'+wire+'::'+row['variant']))
    return rows,dict(source_admission=admission,manifest_sha256=manifest_sha256,
        source_recordings=len(sources),views=len(rows),actor_admitted=False)


def select(rows,seed,step,source_batch=8):
    if source_batch!=8 or type(source_batch) is not int:raise ValueError('eight sources and32 views required')
    groups={}
    for row in rows:groups.setdefault(row['source_id'],[]).append(row)
    for group in groups.values():
        if [r['variant'] for r in group]!=list(VARIANTS):raise ValueError('whole ordered source-view group required')
    chosen=schedule([dict(id=uid) for uid in groups],seed,step,source_batch)
    selected=[r for s in chosen for r in groups[s['id']]]
    return selected,dict(source_ids=[s['id'] for s in chosen],view_ids=[r['id'] for r in selected],
        gate_ids=[r['gate_id'] for r in selected],source_uses=8,view_uses=32,
        request_weight_sum=sum(r['target']['request_weight'] for r in selected))


def bind_behavior(item,request_id,identity,hf_record,vllm_record,policy_id):
    """Call after real processor encoding/bind_span_request, before tensor transport.

    Caller authenticates gate exits and full population separately. Full input
    identity must match both backends, including processed audio and candidates.
    """
    from .span_parity import compare_span_policies
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if (request_id!=digest or identity['policy_id']!=policy_id
            or content_hash(identity['request'])!=content_hash(item['request'].model_input())):
        raise ValueError('encoded policy/request differs from augmented input')
    for record in (hf_record,vllm_record):
        if (record['id']!=item['gate_id'] or record['request_id']!=request_id
                or record['request_identity']!=identity):
            raise ValueError('exact transformed behavior evidence required')
        logp=record['candidate_logprobs']
        if len(logp)!=len(item['request'].questions) or any(len(p)!=len(q.values) for p,q in zip(logp,item['request'].questions,strict=True)):
            raise ValueError('complete transformed candidate support required')
    audit=compare_span_policies(hf_record['candidate_logprobs'],vllm_record['candidate_logprobs'])
    if not audit['passed']:raise ValueError('transformed HF/vLLM candidate parity failed')
    return vllm_record['candidate_logprobs'],audit

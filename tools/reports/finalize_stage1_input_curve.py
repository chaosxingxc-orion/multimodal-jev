"""Full curve audit with policy-independent inputs and policy-bound requests.

The complete original point audit still authenticates every engine request.
Across checkpoints compare actual input evidence, excluding only the request
identifier which intentionally contains the checkpoint policy identity.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import FunctionType

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports.context_isolated_source_audit import IsolatedSourceAuditEngine
from tools.reports import context_audit_optimized_pipeline as optimized
from tools.reports import collect_context_replica_infonce_curve as curve
from tools.reports.prepare_objective_training_candidate import read_bound


def input_identity(bound,proof):
    p=Path(bound['collection'])/'evaluation/inference/input-receipts.json'
    if file_hash(p)!=proof['inference_audit']['input_receipts_sha256']:
        raise ValueError('audited input receipts changed')
    rows=json.loads(p.read_text());policy=proof['result']['policy_id']
    normalized=[]
    for row in rows:
        expected={'id','audio_sha256','request_sha256','candidate_ids','positions','inputs','engine_request_id'}
        if set(row)!=expected:raise ValueError('complete known input evidence required')
        request=dict(policy_id=policy,inputs=row['inputs'],positions=row['positions'])
        digest=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if row['engine_request_id']!=digest:raise ValueError('request not bound to audited checkpoint')
        normalized.append({k:v for k,v in row.items() if k!='engine_request_id'})
    return hashlib.sha256(json.dumps(normalized,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def reconstructor(engine):
    _,point=optimized.optimized_selection(engine)
    source=inspect.getsource(curve.reconstruct)
    before="input_receipts_sha256=proof['inference_audit']['input_receipts_sha256'],"
    if source.count(before)!=1:raise ValueError('curve input control anchor changed')
    source=source.replace(before,'input_receipts_sha256=input_identity(bound,proof),')
    scope=dict(curve.reconstruct.__globals__,point=point,reconstruct_plan=engine.reconstruct,input_identity=input_identity)
    exec(compile(source,'<policy-independent-curve-inputs>','exec'),scope)
    return scope['reconstruct']


def run(c,provenance):
    prior=json.loads(read_bound(c['curve_config']));root=Path(c['output']);root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    spec=dict(prior,output=str(root/'curve'));commit_json(root/'curve-config.json',spec)
    try:
        with IsolatedSourceAuditEngine(cpu_workers=c['cpu_workers'],io_workers=c['io_workers']) as engine:
            result=curve.publication.collect(spec,provenance,reconstructor=reconstructor(engine))
        commit_json(root/'terminal.json',dict(exit_code=0,status=result['status'],npu_executed=False))
        return result
    except BaseException as e:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(e),restart=False));raise


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local);print(json.dumps(run(c,provenance),indent=2))

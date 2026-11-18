"""Shared four-card lifecycle and independent full-selection score audit."""
import argparse
import inspect
import json
from pathlib import Path
from types import FunctionType

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_runtime_profile import PROFILE
from multimodal_jev.span_request_identity import tensor_identity
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.context_replica_macro_rollout import plan_rollouts
from tools.reports import context_train_parity_operations as lifecycle
from tools.reports import audit_context_train_parity as transport
from tools.reports import prepare_context_replica_selection as preparation
from tools.reports import context_replica_selection_client as client
from tools.reports.audit_context_macro_evaluation import audit as audit_macros
from tools.reports.audit_context_replica_training import audit_journal
from tools.jev.context_metric_equivalence import compare as compare_metrics


def read(p):return json.loads(Path(p).read_text())


def validate(c):
    spec=c['selection_package'];preparation.audit(spec,{'source':'selection_lifecycle'})
    fn=FunctionType(lifecycle.validate.__code__,dict(lifecycle.validate.__globals__,audit_package=lambda *a:None))
    result=fn(dict(c,parity_package=spec))
    from tools.reports.context_release_deployment import audit as audit_release
    audit_release(lifecycle.command(c,result[0],result[1],producer=True),result[1])
    return result


def launch(c,provenance):
    if c.get('selection_dispatch_allowed') is not True:raise ValueError('explicit full selection authorization required')
    fn=FunctionType(lifecycle.launch.__code__,dict(lifecycle.launch.__globals__,validate=validate))
    result=fn(dict(c,finite_inference_dispatch_allowed=True),provenance)
    return dict(result,status='selection_collected_independent_audit_required')


def independent_inputs(package,collection,config):
    """Bind emitted marker values to original tensor hashes before using them."""
    package,collection=Path(package),Path(collection)
    local=dict(config)
    for key,name in [('cache_audit','original-cache-audit.json'),('cache_receipts','original-cache-stream.jsonl')]:
        local[key]=dict(config[key],path=str(package/'selection-inputs'/name))
    population,_,candidates=client.inputs(local)
    receipts,proof=client.load_evidence(local,population,
        release_manifest=read(Path(local['release'])/'MANIFEST.json'),
        processor_sha256=config['dual_runtime_identity']['processor_sha256'])
    emitted=read(collection/'evaluation/inference/input-receipts.json')
    if len(emitted)!=9226:raise ValueError('full selection input receipts required')
    expected=[]
    for asset,original,observed in zip(population,receipts,emitted,strict=True):
        positions=observed['positions'];tensors=original['tensors']
        if (len(positions)!=60 or any(type(v) is not int for v in positions)
                or tensor_identity(torch.tensor([positions],dtype=torch.long))!=tensors['positions']):
            raise ValueError('selection marker values differ from independent cache evidence')
        expected.append(dict(asset,candidate_ids=candidates,positions=positions,
            inputs={k:v for k,v in tensors.items() if k!='positions'}))
    selected=client.panel(population,candidates,receipts);lookup={r['id']:r for r in expected}
    selected['input_receipts']=[lookup[a['id']] for a in selected['assets']]
    if (read(package/'selection-inputs/panel.json')!={k:v for k,v in selected.items() if k!='input_receipts'}
            or read(collection/'hf/panel.json')!=selected or read(collection/'evaluation/panel.json')!=selected
            or read(collection/'hf/input-evidence.json')!=proof
            or read(collection/'evaluation/input-evidence.json')!=proof):
        raise ValueError('selection panel/evidence differs from independent reconstruction')
    return population,candidates,expected,selected


def audit_outputs(package,collection):
    package,collection=Path(package),Path(collection)
    m=read(package/'PREPARED.json');c=read(package/'parity-macro.json');pool=c['macro_client']
    population,candidates,receipts,selected=independent_inputs(package,collection,c)
    for role in ('hf','evaluation'):
        if read(collection/role/'terminal.json')!={'exit_code':0}:raise ValueError('successful evaluation phases required')
    if (read(collection/'hf-process/terminal.json')['exit_code']!=0
            or read(collection/'hf/hf-runtime.json')['profile']!=PROFILE):
        raise ValueError('successful owned HF with declared runtime required')
    supervision=read(collection/'supervision/terminal.json')
    if (supervision.get('passed') is not True or supervision.get('actor_started') is not True
            or supervision.get('actor_exit_code')!=0 or supervision.get('forced_cleanup') is not False
            or supervision.get('child_exit_codes')!=[0,0,0]
            or [s['name'] for s in supervision['services']]!=[s['name'] for s in m['inference_services']]
            or any(s.get('exit_code')!=0 for s in supervision['services'])):
        raise ValueError('owned full selection client and services must complete')
    for binding in m['inference_services']:
        name=binding['name'];service=read(package/m['layout']['services'][name]);p=service['policy']
        ready=read(collection/f'service-{name}/ready.json')
        expected=dict(policy_id=c['policy_id'],contract=c['contract'],physical_devices=[binding['device']],
            transport_protocol='bulk_dual_logits_candidate_v1',model_forward_mode=pool['runtime_identity']['profile'],
            max_batch_queries=p['max_batch_queries'],max_batch_tokens=p['max_batch_tokens'],
            entrypoint_sha256=service['pins'][service['source_root']+'/tools/reports/serve_context_dual_candidate.py'])
        if (any(ready.get(k)!=v for k,v in expected.items())
                or read(collection/f'service-{name}/terminal.json')['exit_code']!=0):
            raise ValueError('actual selection service differs: '+name)
    inference=collection/'evaluation/inference';journal=collection/'selection-journal'/pool['attempt_id']
    proof=audit_macros(inference,journal,population=population,candidate_ids=candidates,
        input_receipts=receipts,policy_id=c['policy_id'],runtime_identity=c['dual_runtime_identity'],macro_queries=128)
    dispatch=[]
    for index,entry in enumerate(read(inference/'result.json')['batches']):
        records=[{k:v for k,v in row.items() if k!='id'} for row in read(inference/entry['path'])['predictions']]
        dispatch.append(audit_journal(journal/f'evaluation-{index:08d}',records,policy=c['policy_id'],client=pool,
            planner=lambda costs,width:plan_rollouts(costs,width,replicas=2)))
    records=read(inference/'predictions.json')['records']
    parity=client.parity(read(collection/'hf/reference.json'),records,selected,c['policy_id'])
    if read(collection/'evaluation/parity.json')!=parity:raise ValueError('reported selection parity differs')
    scored=client.score(c['release'],c['release_sha256'],inference/'predictions.json',
        file_hash(inference/'predictions.json'),c['policy_id'],'selection')
    reported=read(collection/'evaluation/result.json');metrics=compare_metrics(scored,reported)
    if (reported['input_receipts_sha256']!=proof['input_receipts_sha256']
            or reported['inference_result_sha256']!=file_hash(inference/'result.json')
            or reported['parity_sha256']!=file_hash(collection/'evaluation/parity.json')):
        raise ValueError('score evidence binding differs')
    return dict(status='complete_replica_infonce_selection_audited',source_step=m['source_step'],
        result=scored,metric_comparison=metrics,inference_audit=proof,parity=parity,dispatch_audit=dispatch,
        training_dispatch_allowed=False,full_population_parity=False,actor_updates=0)


def audit(c,provenance):
    fn=FunctionType(transport.audit.__code__,dict(transport.audit.__globals__,
        audit_package=preparation.audit,audit_outputs=audit_outputs))
    result=fn(dict(c,parity_package=c['selection_package']),provenance)
    result['scope']='Complete 9226 selection point with per-policy HF256 parity; no full curve, TEST or transfer claim'
    return result


def run(c,provenance):
    if c.get('selection_dispatch_allowed') is not True:raise ValueError('explicit full selection authorization required')
    root=Path(c['operations_root'])/'finalization';check_paths({'selection_finalization':root})
    root.mkdir(parents=True,exist_ok=False);commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    try:
        result=launch(c,provenance)
        if result['status']!='selection_collected_independent_audit_required':raise ValueError('full selection collection required')
        stem=Path(c['operations_root'])/c['tag']
        bound=dict(selection_package=c['selection_package'],deployment=c,
            **{k:str(stem)+suffix for k,suffix in (
                ('launch','-launch.json'),('inspect','-inspect.json'),('wait','-wait.json'),
                ('transport','-collected.transport.json'),('collection','-collected'),('archive','-collected.tar'))})
        commit_json(root/'audit-config.json',bound);proof=audit(bound,provenance)
        if proof['status']!='complete_replica_infonce_selection_audited':raise ValueError('independent complete selection audit required')
        commit_json(root/'audit.json',proof)
        commit_json(root/'terminal.json',dict(exit_code=0,status='selection_point_collected_and_audited',source_step=proof['source_step']))
        return proof
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True));raise


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['validate','run','audit'],default='validate');a=p.parse_args()
    c,provenance=load_config(a.config,a.local)
    if a.action=='run':result=run(c,provenance)
    elif a.action=='audit':result=audit(c,provenance)
    else:validate(c);result=dict(status='selection_lifecycle_validated_not_executed',npu_executed=False)
    print(json.dumps(result,indent=2))

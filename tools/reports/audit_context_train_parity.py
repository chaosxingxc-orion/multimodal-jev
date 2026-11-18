"""Independently audit collected endpoint TRAIN HF/batch32 parity.

Reconstruct input identities from the original TRAIN tensor hashes, then replay
the shared macro journal and exact comparison. No scoring, training or dispatch.
"""
import argparse
import json
from pathlib import Path

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_reference_dataset import digest
from multimodal_jev.context_runtime_profile import PROFILE
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.span_request_identity import tensor_identity
from tools.reports.context_cached_storage_operations import command,audit_container
from tools.reports.context_train_parity_inputs import load as load_inputs
from tools.reports.context_transfer_macro_parity import compare
from tools.reports.audit_context_macro_evaluation import audit as audit_macros
from tools.reports.prepare_context_train_parity_job import audit as audit_package
from tools.reports.audit_context_replica_training import audit_journal
from multimodal_jev.context_replica_macro_rollout import plan_rollouts


def read(p):return json.loads(Path(p).read_text())


def audit_dispatch(inference,journal,client,*,policy_id):
    """Rebuild actual rollout32 ownership using the training journal verifier."""
    if (client['max_queries']!=128 or client['rollout_queries']!=32
            or len(client['replicas'])!=2):
        raise ValueError('TRAIN parity requires macro128 rollout32 and two replicas')
    result=read(Path(inference)/'result.json');proofs=[]
    for index,entry in enumerate(result['batches']):
        batch=read(Path(inference)/entry['path'])
        records=[{k:v for k,v in row.items() if k!='id'} for row in batch['predictions']]
        if len(records)!=128:raise ValueError('two full 128-query parity macros required')
        proofs.append(audit_journal(Path(journal)/f'evaluation-{index:08d}',records,
            policy=policy_id,client=client,
            planner=lambda costs,width:plan_rollouts(costs,width,replicas=2)))
    if len(proofs)!=2:raise ValueError('complete 256-query dispatch evidence required')
    return dict(macro_queries=128,rollout_queries=32,replicas=2,macros=proofs,
        ownership_reconstructed=True,engine_internal_batching_verified=False)


def independent_panel(inputs,actual):
    """Authenticate concrete marker values by hashing against independent proof.

    The controller does not require remote cache access. Original input evidence
    fixes every tensor; the returned panel must match the producer byte-for-byte.
    """
    receipts=actual['input_receipts']
    if len(receipts)!=len(inputs['inputs']):raise ValueError('complete parity input receipts required')
    assets=[];expected=[]
    for row,receipt in zip(inputs['inputs'],receipts,strict=True):
        prior=row['receipt'];positions=receipt['positions'];tensors=prior['tensors']
        if (not isinstance(positions,list) or len(positions)!=len(inputs['candidate_ids'])
                or any(type(p)is not int or not 0<=p<tensors['input_ids']['shape'][1] for p in positions)
                or positions!=sorted(set(positions))
                or tensor_identity(torch.tensor([positions],dtype=torch.long))!=tensors['positions']):
            raise ValueError('TRAIN marker values differ from independent evidence')
        asset={k:prior[k] for k in ('id','audio_sha256','request_sha256')}
        assets.append(asset)
        expected.append(dict(asset,candidate_ids=inputs['candidate_ids'],positions=positions,
            inputs={k:v for k,v in tensors.items() if k!='positions'}))
    panel=dict(schema='jev-source-train-runtime-panel-v1',dataset='SLURP_clean_TRAIN60',
        candidate_ids=inputs['candidate_ids'],assets=assets,input_receipts=expected,
        selection_sha256=digest(inputs['panel']),input_manifest_sha256=actual['input_manifest_sha256'],
        labels_used=False)
    if actual!=panel:raise ValueError('TRAIN input panel differs from independent evidence')
    return panel


def audit_outputs(package,collection):
    package=Path(package);collection=Path(collection)
    m=read(package/'PREPARED.json');c=read(package/'parity-macro.json');client=c['macro_client']
    inputs=load_inputs(dict(c['train_inputs'],output=str(package/'parity-inputs')))
    selected=read(collection/'hf/panel.json')
    if selected['input_manifest_sha256']!=c['train_inputs']['manifest_sha256']:
        raise ValueError('TRAIN panel manifest differs')
    selected=independent_panel(inputs,selected)
    if read(collection/'evaluation/panel.json')!=selected:raise ValueError('HF/macro panel differs')
    for role in ('hf','evaluation'):
        if read(collection/role/'terminal.json')!={'exit_code':0}:
            raise ValueError('successful complete parity phases required')
    if read(collection/'hf-process/terminal.json')['exit_code']!=0:
        raise ValueError('owned HF process did not succeed')
    if read(collection/'hf/hf-runtime.json')['profile']!=PROFILE:
        raise ValueError('actual HF runtime profile differs')
    supervision=read(collection/'supervision/terminal.json')
    if (supervision.get('passed') is not True or supervision.get('actor_started') is not True
            or supervision.get('actor_exit_code')!=0 or supervision.get('forced_cleanup') is not False
            or supervision.get('child_exit_codes')!=[0,0,0]
            or [s['name'] for s in supervision['services']]!=[s['name'] for s in m['inference_services']]
            or any(s.get('exit_code')!=0 for s in supervision['services'])):
        raise ValueError('successful owned macro client and services required')
    for binding in m['inference_services']:
        name=binding['name'];s=read(package/m['layout']['services'][name]);p=s['policy']
        ready=read(collection/('service-'+name)/'ready.json')
        expected=dict(policy_id=c['source_policy_id'],contract=c['source_contract'],
            physical_devices=[binding['device']],transport_protocol='bulk_dual_logits_candidate_v1',
            model_forward_mode=client['runtime_identity']['profile'],max_batch_queries=p['max_batch_queries'],
            max_batch_tokens=p['max_batch_tokens'],
            entrypoint_sha256=s['pins'][s['source_root']+'/tools/reports/serve_context_dual_candidate.py'])
        if (any(ready.get(k)!=v for k,v in expected.items())
                or read(collection/('service-'+name)/'terminal.json')['exit_code']!=0):
            raise ValueError('actual parity service startup or terminal differs: '+name)
    inference=collection/'evaluation/inference'
    proof=audit_macros(inference,collection/'parity-journal'/client['attempt_id'],
        population=selected['assets'],candidate_ids=selected['candidate_ids'],
        input_receipts=selected['input_receipts'],policy_id=c['source_policy_id'],
        runtime_identity=client['runtime_identity'],macro_queries=128)
    dispatch=audit_dispatch(inference,collection/'parity-journal'/client['attempt_id'],client,
        policy_id=c['source_policy_id'])
    reference=collection/'hf/reference.json'
    result=compare(read(reference),read(inference/'predictions.json')['records'],selected,policy_id=c['source_policy_id'])
    reported=read(collection/'evaluation/comparison.json')
    if (not result['exact'] or any(reported.get(k)!=v for k,v in result.items())
            or reported.get('training_dispatch_allowed') is not False
            or reported['hf_reference']!={'path':m['remote_root']+'/hf/reference.json','sha256':file_hash(reference)}
            or read(collection/'evaluation/macro-output-audit.json')!=proof
            or reported['macro_output_audit_sha256']!=file_hash(collection/'evaluation/macro-output-audit.json')):
        raise ValueError('independent TRAIN logits/probability or journal comparison differs')
    return dict(status='train_endpoint_macro_parity_outputs_audited',comparison=result,inference_audit=proof,
        dispatch_audit=dispatch,
        input_manifest_sha256=c['train_inputs']['manifest_sha256'],actor_updates=0,
        full_population_parity=False,training_dispatch_allowed=False)


def audit(c,provenance):
    spec=c['parity_package'];package=Path(spec['output']);preparation=audit_package(spec,provenance)
    m=read(package/'PREPARED.json');config=read(package/'config.json')
    argv=command(c['deployment'],m,config,producer=True)
    handle=read(c['launch']);observed=read(c['inspect']);transport=read(c['transport'])
    storage=audit_container(argv,config,handle,observed,producer=True)
    if (handle['manifest_sha256']!=spec['manifest_sha256'] or handle['remote_root']!=m['remote_root']
            or handle['layout']!=m['layout'] or handle['config_sha256']!=file_hash(package/'config.json')
            or handle['services']!={n:file_hash(package/p) for n,p in m['layout']['services'].items()}
            or handle['image']!=observed['Image']
            or read(c['wait'])!=dict(container_id=handle['container_id'],exit_code=0)
            or transport['status']!='transport_complete_independent_audit_required'
            or transport['container_id']!=handle['container_id'] or transport['image']!=observed['Image']
            or transport['remote_root']!=m['remote_root'] or transport['state']!=observed['State']
            or any(transport[k] for k in ('bad_pins','bad_package','launch_binding_mismatches'))):
        raise ValueError('successful exact TRAIN parity transport required')
    collection=Path(c['collection'])
    actual={str(p.relative_to(collection)):file_hash(p) for p in collection.rglob('*') if p.is_file()}
    if (collection.is_symlink() or any(p.is_symlink() for p in collection.rglob('*'))
            or actual!=transport['files'] or file_hash(c['archive'])!=transport['archive_sha256']):
        raise ValueError('complete unchanged TRAIN parity collection required')
    for name in m['layout']['configs']+list(m['layout']['services'].values()):
        if actual.get(name)!=file_hash(package/name):raise ValueError('executed parity config differs: '+name)
    result=audit_outputs(package,collection)
    return dict(result,package_audit=preparation,storage_audit=storage,
        container_id=handle['container_id'],image=observed['Image'],manifest_sha256=spec['manifest_sha256'],
        transport_sha256=file_hash(c['transport']),runtime_provenance_verified=True,
        scope='endpoint318 TRAIN256 native60, macro128/rollout32, two TP1 services; no training or full population claim',
        provenance=provenance)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    result=audit(c,provenance);commit_json(c['output'],result)
    print(json.dumps(result,indent=2))

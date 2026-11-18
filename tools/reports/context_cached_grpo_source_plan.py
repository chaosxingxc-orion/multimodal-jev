"""Bind the registered GRPO curve to actual audited cached-training policies.

No inference or selection. Step zero is the actual Stage1 endpoint used by the
formal run; neither an efficiency endpoint nor the Stage1 selected checkpoint.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_policy_artifacts import read_execution_bundle,verify_native_adapter
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports import context_cached_formal as formal
from tools.reports.context_macro_source_evaluation_plan import bound


def read(path):return json.loads(Path(path).read_text())


def build(c,provenance):
    audit_path=Path(c['training_audit_config'])
    if file_hash(audit_path)!=c['training_audit_config_sha256']:raise ValueError('formal audit configuration changed')
    operations=read(audit_path)
    # Reconstruct container, package, transport, all updates and tensor evidence.
    proof=formal.audit(operations,provenance)
    spec=operations['training_package'];package=Path(spec['output']);collection=Path(operations['collection'])
    training=read(package/'config.json');manifest=read(package/'PREPARED.json')
    if (proof['status']!='formal_cached_reference_training_outputs_audited'
            or len(proof['updates'])!=634 or proof['real_query_exposures']!=81026
            or proof['online_reference_calls']!=0
            or proof['manifest_sha256']!=spec['manifest_sha256']
            or file_hash(package/'PREPARED.json')!=spec['manifest_sha256']
            or proof['transport_sha256']!=file_hash(operations['transport'])):
        raise ValueError('complete independently audited cached GRPO source required')
    formal.declared_curve(training);formal.contract(training)
    mounted=sorted(manifest['training_devices']+[s['device'] for s in manifest['inference_services']])
    contract=formal.expected_contract(training,mounted)
    files=read(operations['transport'])['files']
    parent=Path(spec['parent']['output']);endpoint=read(parent/'initial/endpoint-commit.json')
    if (endpoint['updates']!=318 or endpoint['joint_policy']['policy_id']!=training['source_policy_id']
            or training['reference_dataset']['identity']['policy_id']!=training['source_policy_id']):
        raise ValueError('actual Stage1 endpoint318 initialization and reference required')

    def policy(snapshot,adapter,policy_id):
        bundle=read_execution_bundle(snapshot,expected_policy_id=policy_id,
            expected_contract=training['source_contract'],expected_base_id=training['base_id'])
        verify_native_adapter(adapter,bundle)
        return dict(snapshot=str(snapshot.resolve()),adapter=str(adapter.resolve()),policy_id=policy_id,
            manifest_sha256=file_hash(snapshot/'MANIFEST.json'))

    # The package audit authenticates these initial assets against their origin.
    initial=policy(package/'initial/policy',package/'initial/adapter',training['source_policy_id'])
    points=[dict(step=0,source_stage='infonce',source_step=318,**initial)]
    for step in formal.STEPS[1:]:
        commit_path=bound(collection,files,f'run/checkpoint-{step:04d}-committed.json')
        commit=read(commit_path)
        publication=bound(collection,files,f'run/policy-{step:04d}-committed.json')
        if (commit['updates']!=step or commit['contract']!=contract
                or commit['joint_policy_directory']!=f'policy-{step:04d}'
                or not isinstance(commit.get('rank_shards'),dict) or len(commit['rank_shards'])!=2
                or {k:v for k,v in commit.items() if k!='rank_shards'}!=read(publication)):
            raise ValueError('registered checkpoint differs from actual policy publication or training contract')
        path=bound(collection,files,f'run/policy-{step:04d}/MANIFEST.json')
        if file_hash(path)!=commit['joint_policy_manifest_sha256']:raise ValueError('committed policy manifest differs')
        item=policy(path.parent,collection/f'run/adapter-{step:04d}',commit['joint_policy']['policy_id'])
        points.append(dict(step=step,source_stage='grpo',source_step=step,
            commit=dict(path=str(commit_path),sha256=file_hash(commit_path)),**item))
    experiment=dict(release_sha256=training['release_sha256'],selection_recordings=9226,
        task_stage={'updates':634},evaluation=dict(checkpoints=list(formal.STEPS),primary='query_macro_set_f1',
            secondary=['top1_event_accuracy','top1_macro_f1','candidate_binary_nll','candidate_brier',
                'set_precision','set_recall','exact_set','mean_selected_count'],set_threshold=.5))
    return dict(schema='jev-cached-grpo-source-evaluation-plan-v1',
        status='audited_source_policies_bound_evaluation_pending',algorithm=training['algorithm'],
        experiment=experiment,initial_policy_id=initial['policy_id'],checkpoints=points,
        source_operations=operations,source_audit=proof,training_config_sha256=file_hash(package/'config.json'),
        source_contract=training['source_contract'],base_id=training['base_id'],
        runtime_identity=training['dual_runtime_identity'],
        stage1_reference_endpoint=dict(step=318,policy_id=initial['policy_id']),
        reference_endpoint=dict(step=318,policy_id=initial['policy_id']),
        grpo_endpoint=dict(step=634,policy_id=points[-1]['policy_id']),
        reference_manifest_sha256=training['reference_dataset']['manifest_sha256'],
        stage_lora_transition=training['stage_lora_transition'],
        target_tuning_allowed=False,model_selection_performed=False,config_provenance=provenance,
        scope='Actual full cached-GRPO policies only; no selection, TEST or transfer scores')


def reconstruct(c):
    if file_hash(c['source_plan'])!=c['source_plan_sha256']:raise ValueError('GRPO source plan changed')
    plan=read(c['source_plan'])
    actual=build(c['source_finalization'],plan['config_provenance'])
    if actual!=plan:raise ValueError('GRPO source plan differs from independent training reconstruction')
    return actual


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    result=build(c,provenance);commit_json(c['output'],result)
    print(json.dumps(dict(status=result['status'],points=len(result['checkpoints']),evaluation_complete=False)))

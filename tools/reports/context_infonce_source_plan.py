"""Bind full Stage1 source-selection curve to audited actual checkpoints.

No target data or model selection. Reuses the existing metric/curve protocol;
the Stage2 reference is always the fixed Stage1 endpoint, never selected here.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_policy_artifacts import read_execution_bundle
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports.audit_context_infonce_training import checkpoint_steps,expected_contract
from tools.reports.context_infonce_formal import audit


def evaluation_protocol(training):
    steps=checkpoint_steps(training)
    if training.get('experiment_kind')!='formal_infonce_source':
        raise ValueError('full formal InfoNCE source required')
    if training['selection_rule']!='highest_set_f1_then_lowest_binary_nll_then_earliest_step':
        raise ValueError('preregistered source selection rule required')
    layout_bound='stage1_layout' in training
    primary='top1_macro_f1' if layout_bound else 'query_macro_set_f1'
    alternate='query_macro_set_f1' if layout_bound else 'top1_macro_f1'
    return dict(release_sha256=training['release_sha256'],selection_recordings=9226,
        task_stage=dict(updates=training['stop_update']),
        evaluation=dict(checkpoints=[0,*steps],primary=primary,
            secondary=['top1_event_accuracy',alternate,'candidate_binary_nll','candidate_brier',
                       'set_precision','set_recall','exact_set','mean_selected_count'],set_threshold=.5))


def build(config,provenance):
    result=audit(config)
    ops,_=load_config(config['operations_config'],config['operations_local'])
    package=Path(ops['package']);c=json.loads((package/'config.json').read_text())
    root=Path(ops['operations_root']);collection=root/(ops['tag']+'-collected')
    transport=json.loads((root/(ops['tag']+'-collected.transport.json')).read_text())
    def bound(name):
        p=collection/name
        if transport['files'].get(name)!=file_hash(p):raise ValueError('changed source policy artifact')
        return p
    def point(step,snapshot,policy_id):
        bundle=read_execution_bundle(snapshot,expected_policy_id=policy_id,
            expected_contract=c['source_contract'],expected_base_id=c['base_id'])
        return dict(step=step,snapshot=str(Path(snapshot).resolve()),policy_id=bundle['policy_id'],
                    manifest_sha256=file_hash(Path(snapshot)/'MANIFEST.json'))
    points=[point(0,config['initial_policy'],c['source_policy_id'])]
    protocol=evaluation_protocol(c)
    for step in protocol['evaluation']['checkpoints'][1:]:
        commit_path=bound(f'run/checkpoint-{step:04d}-committed.json');commit=json.loads(commit_path.read_text())
        if commit['updates']!=step or commit['contract']!=expected_contract(c):
            raise ValueError('wrong checkpoint origin or objective')
        manifest=bound(f'run/policy-{step:04d}/MANIFEST.json')
        if file_hash(manifest)!=commit['joint_policy_manifest_sha256']:
            raise ValueError('policy differs from committed checkpoint')
        points.append(dict(point(step,manifest.parent,commit['joint_policy']['policy_id']),
            commit=dict(path=str(commit_path),sha256=file_hash(commit_path))))
    layout={}
    if 'stage1_layout' in c:
        layout['stage1_layout']=expected_contract(c)['stage1_layout']
    if 'design' in ops:
        from multimodal_jev.stage_experiment_contract import source_curve
        design_path=Path(ops['design'])
        if file_hash(design_path)!=json.loads((package/'PREPARED.json').read_bytes())['design_sha256']:
            raise ValueError('formal source design changed')
        design=json.loads(design_path.read_bytes())
        curve=source_curve(design)
        if (design['stage1']['batch_config']!=c['batch_config'] or curve['steps']!=protocol['evaluation']['checkpoints']
                or curve['world_size']!=result['world_size'] or curve['real_exposures']!=result['real_exposures']):
            raise ValueError('audited run differs from configured source curve')
        protocol['selection_recordings']=curve['selection_recordings']
        layout['source_curve']=curve
    return dict(schema='jev-infonce-source-evaluation-plan-v1',**layout,
        status='audited_source_policies_bound_evaluation_pending',algorithm='candidate_infonce',
        experiment=protocol,checkpoints=points,initial_policy_id=c['source_policy_id'],
        source_operations=ops,source_audit=result,training_config_sha256=file_hash(package/'config.json'),
        source_contract=c['source_contract'],base_id=c['base_id'],
        reference_endpoint=dict(step=c['reference_endpoint'],policy_id=points[-1]['policy_id']),
        stage_lora_transition=c.get('stage_lora_transition'),target_tuning_allowed=False,
        model_selection_performed=False,config_provenance=provenance,
        scope='Full source selection plan from independently audited InfoNCE policies; no scores or transfer claim')


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    result=build(c,provenance);commit_json(c['output'],result)
    print(json.dumps(dict(status=result['status'],points=len(result['checkpoints']))))

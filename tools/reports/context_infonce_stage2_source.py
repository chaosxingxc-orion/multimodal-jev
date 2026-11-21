"""Bind the audited InfoNCE endpoint to a portable Stage2 policy/LoRA package.

No selection-best substitution, base-weight copy, optimizer-state carryover or
NPU admission. The public prepare/audit paths independently reconstruct origin.
"""
import argparse
import json
from pathlib import Path
import shutil

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_policy_artifacts import (
    read_execution_bundle, export_native_adapter, verify_native_adapter,
)
from multimodal_jev.context_stage_lora import bind_design, admit_stage2
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.context_infonce_selection import reconstruct_plan, validate_plan
from tools.reports.audit_context_infonce_training import expected_contract
from tools.reports.context_cached_reference_trainer import BASELINE_OBJECTIVE, validate_objective


def read(path):
    return json.loads(Path(path).read_text())


def binding(c, plan):
    validate_plan(plan)
    design_path = Path(c['design'])
    if file_hash(design_path) != c['design_sha256']:
        raise ValueError('two-stage design changed')
    design = read(design_path); stage = design['stage2']
    from multimodal_jev.stage_experiment_contract import reference_batch, source_curve
    curve=source_curve(design); batch=reference_batch(design)
    endpoint_step=curve['steps'][-1]
    if (plan['reference_endpoint']['step']!=endpoint_step or plan['source_audit']['world_size']!=curve['world_size']
            or plan['experiment']['evaluation']['checkpoints']!=curve['steps']
            or stage['optimizer_reset'] is not True or design['reference']['updates_during_stage2'] is not False
            or design['reference']['mode']!='complete_train_readonly_result_cache'):
        raise ValueError('full authenticated source and frozen reference required')
    extra={}
    if 'objective_config' in stage:
        from multimodal_jev.source_snapshot import regular
        root=Path(__file__).resolve().parents[2]
        ref=stage['objective_config']; config=regular(root,ref['path'])
        if file_hash(config)!=ref['sha256']:
            raise ValueError('objective design changed')
        payload=read(config)
        if file_hash(regular(root,payload['objective_source']))!=payload['objective_sha256']:
            raise ValueError('objective implementation changed')
        extra['objective_config']=ref
        objective={key:stage[key] for key in ('algorithm','arm','group_size','clip_ratio','advantage_std_floor','learning_rate')}
    else:
        objective = {k:stage[k] for k in BASELINE_OBJECTIVE}
        validate_objective(dict(stage2_objective=objective,algorithm=stage['algorithm'],dual_arm=stage['arm']))
    endpoint = plan['checkpoints'][-1]
    training_path = Path(plan['source_operations']['package'])/'config.json'
    if file_hash(training_path) != plan['training_config_sha256']:
        raise ValueError('formal source configuration changed')
    training = read(training_path)
    for name,key in [('batch_config','batch_config'),('instruction_objective','instruction_objective'),
                     ('lr_schedule','lr_schedule')]:
        if training[name] != design['stage1'][key]:
            raise ValueError('design differs from actual Stage1: '+name)
    commit_path = Path(endpoint['commit']['path'])
    if file_hash(commit_path) != endpoint['commit']['sha256']:
        raise ValueError('Stage1 endpoint commit changed')
    commit = read(commit_path)
    if (commit['updates'] != endpoint_step or commit['contract'] != expected_contract(training)
            or commit['joint_policy']['policy_id'] != endpoint['policy_id']
            or commit['joint_policy_manifest_sha256'] != endpoint['manifest_sha256']):
        raise ValueError('actual committed full InfoNCE endpoint required')
    layout={}
    if 'stage1_layout' in training or 'expected_stage1_layout' in c:
        from multimodal_jev.stage2_dataset_release import require_endpoint_layout
        layout['stage1_layout']=require_endpoint_layout(commit['contract'],c.get('expected_stage1_layout'))
    snapshot = Path(endpoint['snapshot'])
    if file_hash(snapshot/'MANIFEST.json') != endpoint['manifest_sha256']:
        raise ValueError('endpoint snapshot changed')
    bundle = read_execution_bundle(snapshot,expected_policy_id=endpoint['policy_id'],
        expected_contract=plan['source_contract'],expected_base_id=plan['base_id'])
    transition = bind_design(design,bundle['contract'])
    if transition != plan['stage_lora_transition']:
        raise ValueError('stage LoRA transition differs from preregistered training')
    admit_stage2(transition,bundle['contract'])
    result = dict(schema='jev-infonce-stage2-source-v1',source_step=endpoint_step,
        source_policy_id=bundle['policy_id'],reference_policy_id=bundle['policy_id'],
        base_id=plan['base_id'],source_contract=bundle['contract'],
        source_manifest_sha256=endpoint['manifest_sha256'],
        source_commit_sha256=endpoint['commit']['sha256'],
        source_plan_sha256=c['source_plan_sha256'],design_sha256=c['design_sha256'],
        release_sha256=training['release_sha256'],stage_lora_transition=transition,
        stage2_objective=objective,batch_config=batch,
        reset_optimizer_for_task_stage=True,reference_mode='complete_train_readonly_result_cache',
        selection_best_used=False,training_dispatch_allowed=False,**layout,**extra)
    return result, bundle, snapshot


def prepare(c, provenance, *, reconstruct=None):
    plan = (reconstruct or reconstruct_plan)(c)
    bound, bundle, snapshot = binding(c,plan)
    output = Path(c['output'])
    for source in (snapshot,Path(plan['source_operations']['package']),Path(c['source_plan']).parent):
        if output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(output.resolve()):
            raise ValueError('new independent source package required')
    check_paths({'stage2_source_package':output})
    output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(snapshot,output/'policy')
    export_native_adapter(output/'adapter',bundle)
    verify_native_adapter(output/'adapter',bundle)
    commit_json(output/'binding.json',bound)
    for source,name in [(c['source_plan'],'source-plan.json'),(c['design'],'design.json'),
                        (plan['checkpoints'][-1]['commit']['path'],'endpoint-commit.json')]:
        shutil.copyfile(source,output/name)
    manifest = dict(status='infonce_stage2_source_prepared_only',source_policy_id=bundle['policy_id'],
        training_dispatch_allowed=False,numerical_admission=False,provenance=provenance,
        files={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file()})
    commit_json(output/'PACKAGE.json',manifest)
    return dict(status=manifest['status'],manifest_sha256=file_hash(output/'PACKAGE.json'),
                source_policy_id=bundle['policy_id'],source_step=bound['source_step'],training_dispatch_allowed=False)


def audit(c, provenance, *, reconstruct=None):
    root=Path(c['output']); manifest=root/'PACKAGE.json'
    if root.is_symlink() or file_hash(manifest)!=c['manifest_sha256']:
        raise ValueError('source package manifest differs')
    m=read(manifest)
    actual={str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file() and p!=manifest}
    if (any(p.is_symlink() for p in root.rglob('*')) or actual!=m['files']
            or m['status']!='infonce_stage2_source_prepared_only'
            or m['training_dispatch_allowed'] is not False or m['numerical_admission'] is not False):
        raise ValueError('complete immutable source-only package required')
    plan=(reconstruct or reconstruct_plan)(c); expected,_,_=binding(c,plan)
    if (read(root/'binding.json')!=expected or m['source_policy_id']!=expected['source_policy_id']
            or file_hash(root/'source-plan.json')!=c['source_plan_sha256']
            or file_hash(root/'design.json')!=c['design_sha256']
            or file_hash(root/'endpoint-commit.json')!=expected['source_commit_sha256']):
        raise ValueError('source package differs from independent origin')
    bundle=read_execution_bundle(root/'policy',expected_policy_id=expected['source_policy_id'],
        expected_contract=expected['source_contract'],expected_base_id=expected['base_id'])
    if file_hash(root/'policy/MANIFEST.json')!=expected['source_manifest_sha256']:
        raise ValueError('copied endpoint manifest differs')
    verify_native_adapter(root/'adapter',bundle)
    allowed={'binding.json','source-plan.json','design.json','endpoint-commit.json'}
    if any(name not in allowed and not name.startswith(('policy/','adapter/')) for name in actual):
        raise ValueError('source package contains unexpected assets')
    return dict(status='infonce_stage2_source_independently_audited',
        manifest_sha256=c['manifest_sha256'],source_policy_id=bundle['policy_id'],source_step=expected['source_step'],
        optimizer_state_copied=False,base_weights_copied=False,
        training_dispatch_allowed=False,npu_executed=False,provenance=provenance)


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','audit'],required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    result=prepare(c,provenance) if a.action=='prepare' else audit(c,provenance)
    if a.action=='audit':commit_json(c['audit_output'],result)
    print(json.dumps(result,indent=2))

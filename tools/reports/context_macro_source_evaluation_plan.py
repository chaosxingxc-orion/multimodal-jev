"""Bind a full source evaluation curve to independently audited formal GRPO.

No inference, score selection or target access. All planned snapshots are the
actual BF16 LoRA/head policies from one formal run and its common Stage1 start.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_policy_artifacts import read_execution_bundle, verify_native_adapter
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.context_macro_formal_candidate import contract, STEPS
from tools.reports.context_macro_formal_operations import audit_collected
from tools.reports.run_context_batch_probe_candidate import verify


def bound(root, files, name):
    relative=Path(name);root=Path(root);path=root/relative
    if (relative.is_absolute() or '..' in relative.parts or path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
            or files.get(name)!=file_hash(path)):
        raise ValueError('changed or unbound source curve asset: '+name)
    return path


def build(operations, admission):
    package=Path(operations['prepared'])/'package'
    base=Path(operations['operations_root']);collection=base/'collected'
    if (admission['status']!='formal_dp2_outputs_audited'
            or admission['files_sha256']!=operations['manifest_sha256']
            or verify(package)!=operations['manifest_sha256']
            or admission['transport_sha256']!=file_hash(base/'transport.json')
            or len(admission['updates'])!=634):
        raise ValueError('complete independently audited formal GRPO source required')
    training=json.loads((package/'training.json').read_text())
    formal=contract(training)
    transport=json.loads((base/'transport.json').read_text());files=transport['files']
    def policy(snapshot,adapter,policy_id):
        bundle=read_execution_bundle(snapshot,expected_policy_id=policy_id,
            expected_contract=training['source_contract'],expected_base_id=training['base_id'])
        verify_native_adapter(adapter,bundle)
        return dict(snapshot=str(snapshot),adapter=str(adapter),policy_id=policy_id,
            manifest_sha256=file_hash(snapshot/'MANIFEST.json'))
    initial=policy(package/'policy',package/'adapter',training['source_policy_id'])
    points=[dict(step=0,source_stage='instruction',source_step=634,**initial)]
    for step in STEPS[1:]:
        name=f'run/checkpoint-{step:04d}-committed.json'
        commit_path=bound(collection,files,name);commit=json.loads(commit_path.read_text())
        if (commit['updates']!=step or commit['contract'].get('formal_experiment')!=formal
                or commit['contract']['source_policy_id']!=initial['policy_id']
                or commit['contract']['algorithm']!=training['algorithm']
                or commit['contract']['release_sha256']!=training['release_sha256']
                or commit['joint_policy_directory']!=f'policy-{step:04d}'):
            raise ValueError('source checkpoint step, origin or algorithm differs')
        directory='run/'+commit['joint_policy_directory']
        manifest=bound(collection,files,directory+'/MANIFEST.json')
        if file_hash(manifest)!=commit['joint_policy_manifest_sha256']:
            raise ValueError('committed source policy identity differs')
        # read_execution_bundle validates every asset and tensor, not only hashes.
        point=policy(manifest.parent,collection/f'run/adapter-{step:04d}',commit['joint_policy']['policy_id'])
        points.append(dict(step=step,source_stage='task',source_step=step,
            commit=dict(path=str(commit_path),sha256=file_hash(commit_path)),**point))
    experiment=dict(release_sha256=training['release_sha256'],selection_recordings=9226,
        task_stage=dict(updates=634),evaluation=dict(checkpoints=list(STEPS),
            primary='query_macro_set_f1',secondary=['top1_event_accuracy','top1_macro_f1',
                'candidate_binary_nll','candidate_brier'],set_threshold=.5))
    return dict(schema='jev-macro-source-evaluation-plan-v1',status='bound_full_curve_pending_evaluation',
        algorithm=training['algorithm'],experiment=experiment,initial_policy_id=initial['policy_id'],
        source_operations=operations,source_audit=admission,checkpoints=points,
        training_config_sha256=file_hash(package/'training.json'),
        runtime_identity=training['dual_runtime_identity'],target_tuning_allowed=False,
        model_selection_performed=False,
        scope='Actual audited formal-run policies bound for 11 full selection evaluations; '
              'no selection scores, target evaluation or new Stage1 provenance certification.')


def prepare(config,provenance):
    operations,_=load_config(config['source_operations_config'],config['source_operations_local'])
    # Always reconstruct tensor, transport and actual container evidence.
    admission=audit_collected(operations)
    plan=build(operations,admission)
    path=Path(config['output']);check_paths({'source_evaluation_plan':path})
    path.parent.mkdir(parents=True,exist_ok=True)
    plan['config_provenance']=provenance;commit_json(path,plan)
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    args=parser.parse_args();config,provenance=load_config(args.config,args.local)
    import torch
    torch.set_num_threads(1)
    print(json.dumps(prepare(config,provenance),indent=2))

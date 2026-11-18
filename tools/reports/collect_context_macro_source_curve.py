"""Re-audit all 11 source evaluations and freeze policy roles for transfer.

Consumes only formal source training and SLURP selection collections. Never
opens target examples/scores and never selects from an incomplete curve.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.jev.context_selection_curve import summarize
from tools.reports.context_macro_formal_operations import audit_collected as audit_training
from tools.reports.context_macro_source_evaluation_plan import build
from tools.reports.context_source_point_macro_operations import audit_collected as audit_point
from tools.reports.evaluate_context_transfer_macro import source_binding


def read(path):
    return json.loads(Path(path).read_text())


def reconstruct(config, *, source_reconstructor=None, point_auditor=None):
    """Read-only independent reconstruction, shared by publication and deployment."""
    if file_hash(config['source_plan'])!=config['source_plan_sha256']:
        raise ValueError('source plan changed')
    plan=read(config['source_plan']);source_ops=plan['source_operations']
    if source_reconstructor is None:
        training=audit_training(source_ops)
        actual=build(source_ops,training)
    else:
        actual=source_reconstructor(config)
        training=actual['source_audit']
    point_auditor=point_auditor or audit_point
    if {k:v for k,v in plan.items() if k!='config_provenance'}!={k:v for k,v in actual.items() if k!='config_provenance'}:
        raise ValueError('source plan differs from independently audited training')
    steps=[point['step'] for point in actual['checkpoints']]
    entries=config['evaluations']
    if ([item['step'] for item in entries]!=steps
            or any(type(item['step']) is not int for item in entries)):
        raise ValueError('complete ordered 11-point source evaluation configuration required')
    points=[];proofs=[];controls=None
    for item,checkpoint in zip(entries,actual['checkpoints'],strict=True):
        operations,_=load_config(item['config'],item['local'])
        # Full archive/runtime/inference audit and independent native rescoring.
        audit=point_auditor(operations)
        package=Path(operations['prepared'])/'package'
        receipt=read(package.parent/'PREPARED.json')
        evaluation=read(package/'evaluation.json')
        if (audit['status']!='complete_source_point_macro_selection_audited'
                or read(package/'SOURCE-PLAN.json')!=actual
                or receipt['source_step']!=item['step']
                or receipt['policy_id']!=checkpoint['policy_id']
                or receipt['parent_manifest_sha256']!=source_ops['manifest_sha256']
                or file_hash(package/'policy/MANIFEST.json')!=checkpoint['manifest_sha256']
                or audit['result']['policy_id']!=checkpoint['policy_id']
                or audit['files_sha256']!=operations['manifest_sha256']):
            raise ValueError('evaluated policy, step or formal source differs')
        matched=dict(runtime=evaluation['macro_client']['runtime_identity'],image=audit['image'],
            contract=evaluation['contract'],base_id=evaluation['base_id'],
            release_sha256=evaluation['release_sha256'],macro_queries=evaluation['macro_queries'],
            audio_token_id=evaluation['audio_token_id'],
            input_receipts_sha256=file_hash(package/'evaluation-inputs/input-receipts.json'),
            cache_audit_sha256=evaluation['cache_audit']['sha256'],
            cache_receipts_sha256=evaluation['cache_receipts']['sha256'])
        matched['service_policies']=[{k:v for k,v in read(package/f'service-{i}.json')['policy'].items()
            if k not in ('policy_id','snapshot','adapter')} for i in range(2)]
        if controls is None:controls=matched
        elif matched!=controls:raise ValueError('source curve runtime, processor or input evidence differs')
        points.append(dict(step=item['step'],result=audit['result']))
        proofs.append(dict(step=item['step'],operations=operations,audit=audit))
    curve=summarize(points,actual['experiment'],initial_policy_id=actual['initial_policy_id'])
    if not curve['complete']:raise ValueError('incomplete source curve cannot select a model')
    return dict(plan=actual,training=training,points=points,proofs=proofs,controls=controls,curve=curve)


def collect(config,provenance):
    checked=reconstruct(config)
    actual=checked['plan'];training=checked['training'];points=checked['points']
    proofs=checked['proofs'];controls=checked['controls'];curve=checked['curve']
    steps=[p['step'] for p in points]
    output=Path(config['output'])
    check_paths({'source_curve':output});output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        path=output/name;commit_json(path,value)
        return dict(path=str(path.resolve()),sha256=file_hash(path))
    try:
        bound_points=[dict(step=p['step'],result=save(f"score-{p['step']:04d}.json",p['result'])) for p in points]
        curve_entry=save('curve.json',curve)
        audit_entry=save('audits.json',dict(source_plan=config['source_plan'],
            source_plan_sha256=config['source_plan_sha256'],training=training,evaluations=proofs,
            matched_controls=controls,config_provenance=provenance))
        source=dict(schema='jev-macro-transfer-source-evaluation-v1',algorithm=actual['algorithm'],
            experiment=actual['experiment'],initial_policy_id=actual['initial_policy_id'],
            selected_checkpoint=curve['selected_checkpoint'],target_tuning_allowed=False,
            points=bound_points,curve=curve_entry,source_audits=audit_entry)
        source_entry=save('source-evaluation.json',source)
        roles=dict(common_Stage1=dict(step=0,policy_id=actual['initial_policy_id']),
            source_selected_new_GRPO=curve['selected_checkpoint'],
            fixed_endpoint_new_GRPO=dict(step=steps[-1],policy_id=points[-1]['result']['policy_id']))
        bindings={role:source_binding(dict(source_evaluation=source_entry,policy_role=role,
                                          policy_id=value['policy_id'])) for role,value in roles.items()}
        result=dict(status='complete_source_curve_audited_and_roles_frozen',source_evaluation=source_entry,
            roles=bindings,selected_checkpoint=curve['selected_checkpoint'],
            points=len(points),source_runtime_provenance_verified=True,target_inference_performed=False,
            scope='Complete independently audited source selection curve and role binding; '
                  'no source TEST, target transfer, cross-task or statistical significance claim.')
        save('result.json',result);save('terminal.json',dict(exit_code=0))
        return result
    except BaseException as exc:
        save('terminal.json',dict(exit_code=1,error=repr(exc)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    args=parser.parse_args();config,provenance=load_config(args.config,args.local)
    import torch
    torch.set_num_threads(1)
    print(json.dumps(collect(config,provenance),indent=2))

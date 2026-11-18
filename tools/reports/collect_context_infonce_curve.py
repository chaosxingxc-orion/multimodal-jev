"""Audit the full InfoNCE selection curve without relabeling it as GRPO.

The source-selected checkpoint is reported separately from the fixed endpoint
that initializes Stage2 and produces its reference cache.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.collect_context_macro_source_curve import reconstruct
from tools.reports.context_infonce_selection import reconstruct_plan, audit_collected


def collect(config,provenance,*,reconstructor=None):
    checked=(reconstructor or reconstruct)(config,source_reconstructor=reconstruct_plan,point_auditor=audit_collected)
    plan=checked['plan'];curve=checked['curve']
    output=Path(config['output']);check_paths({'infonce_curve':output})
    output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        path=output/name;commit_json(path,value)
        return dict(path=str(path.resolve()),sha256=file_hash(path))
    try:
        scores=[dict(step=p['step'],result=save(f"score-{p['step']:04d}.json",p['result']))
                for p in checked['points']]
        curve_entry=save('curve.json',curve)
        audit_entry=save('audits.json',dict(source_plan=config['source_plan'],
            source_plan_sha256=config['source_plan_sha256'],training=checked['training'],
            evaluations=checked['proofs'],matched_controls=checked['controls'],config_provenance=provenance))
        roles=dict(initial=dict(step=0,policy_id=plan['initial_policy_id']),
            source_selected=curve['selected_checkpoint'],
            stage2_initial_and_reference=plan['reference_endpoint'])
        source=save('source-evaluation.json',dict(schema='jev-infonce-source-evaluation-v1',
            algorithm='candidate_infonce',experiment=plan['experiment'],points=scores,curve=curve_entry,
            source_audits=audit_entry,roles=roles,target_tuning_allowed=False))
        result=dict(status='complete_infonce_source_curve_audited',source_evaluation=source,roles=roles,
            points=len(scores),source_runtime_provenance_verified=True,target_inference_performed=False,
            scope='Full InfoNCE source selection only; not GRPO improvement, probability calibration, TEST or transfer evidence')
        save('result.json',result);save('terminal.json',dict(exit_code=0))
        return result
    except BaseException as exc:
        save('terminal.json',dict(exit_code=1,error=repr(exc)));raise


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    args=parser.parse_args();c,p=load_config(args.config,args.local)
    print(json.dumps(collect(c,p),indent=2))

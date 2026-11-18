"""Reconstruct the complete InfoNCE curve from replica-protocol collections.

Reuse source provenance, per-point transport/numeric audit, the native scorer,
selection rule and publication schema. No target data or optimizer updates.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.jev.context_selection_curve import summarize
from tools.reports.context_infonce_selection import reconstruct_plan
from tools.reports import context_replica_selection_operations as point
from tools.reports import collect_context_infonce_curve as publication


def read(path):return json.loads(Path(path).read_text())


def reconstruct(c,**_):
    plan=reconstruct_plan(c)
    entries=c['evaluations'];steps=[p['step'] for p in plan['checkpoints']]
    if ([e['step'] for e in entries]!=steps or any(type(e['step']) is not int for e in entries)):
        raise ValueError('complete ordered registered InfoNCE curve required')
    points=[];proofs=[];controls=None;seen=set()
    for entry,checkpoint in zip(entries,plan['checkpoints'],strict=True):
        path=Path(entry['audit_config'])
        if file_hash(path)!=entry['audit_config_sha256']:raise ValueError('point audit configuration changed')
        bound=read(path);spec=bound['selection_package'];root=Path(spec['output'])
        identity=str(root.resolve())
        if identity in seen:raise ValueError('duplicate selection package')
        seen.add(identity)
        proof=point.audit(bound,{'source':'complete_replica_infonce_curve'})
        m=read(root/'PREPARED.json');evaluation=read(root/'parity-macro.json')
        if (proof['status']!='complete_replica_infonce_selection_audited'
                or proof['runtime_provenance_verified'] is not True
                or proof['actor_updates']!=0 or proof['parity']['exact'] is not True
                or proof['source_step']!=entry['step'] or spec['source_step']!=entry['step']
                or spec['source_plan_sha256']!=c['source_plan_sha256']
                or read(root/'SOURCE-PLAN.json')!=plan or m['source_step']!=entry['step']
                or m['source_policy_id']!=checkpoint['policy_id']
                or file_hash(root/'initial/policy/MANIFEST.json')!=checkpoint['manifest_sha256']
                or proof['manifest_sha256']!=spec['manifest_sha256']
                or proof['result']['policy_id']!=checkpoint['policy_id']):
            raise ValueError('actual registered policy, source and audited result required')
        matched=dict(parent=spec['parent'],image=proof['image'],
            runtime=evaluation['dual_runtime_identity'],contract=evaluation['contract'],
            base_id=evaluation['base_id'],release_sha256=evaluation['release_sha256'],
            macro_queries=evaluation['macro_queries'],rollout_queries=evaluation['macro_client']['rollout_queries'],
            audio_token_id=evaluation['audio_token_id'],
            selection_client_sha256=file_hash(root/'selection-client.py'),
            panel_sha256=file_hash(root/'selection-inputs/panel.json'),
            input_receipts_sha256=proof['inference_audit']['input_receipts_sha256'],
            cache_audit_sha256=evaluation['cache_audit']['sha256'],
            cache_receipts_sha256=evaluation['cache_receipts']['sha256'])
        matched['services']=[{k:v for k,v in read(root/m['layout']['services'][s['name']])['policy'].items()
            if k not in ('policy_id','snapshot','adapter')} for s in m['inference_services']]
        matched['service_execution']=[{k:v for k,v in read(root/m['layout']['services'][s['name']]).items()
            if k not in ('policy','pins','socket_path','exchange','source_root','source_policy','dual_runtime_manifest')}
            for s in m['inference_services']]
        if controls is None:controls=matched
        elif controls!=matched:raise ValueError('curve model runtime, client or inputs differ')
        points.append(dict(step=entry['step'],result=proof['result']))
        proofs.append(dict(step=entry['step'],operations=bound,audit=proof))
    curve=summarize(points,plan['experiment'],initial_policy_id=plan['initial_policy_id'])
    if not curve['complete']:raise ValueError('incomplete curve cannot select a model')
    return dict(plan=plan,training=plan['source_audit'],points=points,proofs=proofs,controls=controls,curve=curve)


def collect(c,provenance):
    return publication.collect(c,provenance,reconstructor=reconstruct)


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    print(json.dumps(collect(c,provenance),indent=2))

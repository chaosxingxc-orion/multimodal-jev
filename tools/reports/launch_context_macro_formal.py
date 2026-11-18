"""Admit and launch fresh full-source training using the existing DP2 lifecycle.

Reconstruct complete efficiency evidence and actual tail acceptance first.
No automatic restart/resume; sparse NPU recovery remains a separate obligation.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports import context_macro_formal_operations as formal
from tools.reports.context_macro_training_operations import run_services
from tools.reports.context_macro_tail_operations import audit_collected as audit_tail
from tools.reports.prepare_context_macro_formal import efficiency_choice, audit_schedule, FORMAL_SOURCES
from tools.reports.prepare_context_dual_replay import relocate
from tools.reports.run_context_batch_probe_candidate import verify


def read(p):return json.loads(Path(p).read_text())


def normalized_training(t,root):
    t=relocate(t,root,'/matched')
    external_pins={p:h for p,h in t['pins'].items() if not Path(p).is_relative_to('/matched/package')}
    for key in ('experiment_kind','stop_update','engineering_prefix','training_dispatch_allowed',
                'checkpoint_interval','evaluation_steps','seed','pins'):
        t.pop(key,None)
    t['dual_runtime_identity'].pop('source_manifest_sha256')
    t['macro_client']['runtime_identity'].pop('source_manifest_sha256')
    t['macro_client'].pop('attempt_id')
    t['external_pins']=external_pins
    return t


def source_inventory(package):
    return {str(p.relative_to(package)):file_hash(p) for p in Path(package).rglob('*.py')}


def admit(c):
    ops=c['operations'];receipt=formal.validate(ops)
    package=Path(ops['prepared'])/'package';t=read(package/'training.json');sealed=read(package/'FORMAL.json')
    if file_hash(package/'FORMAL.json')!=receipt['formal_evidence_sha256']:
        raise ValueError('formal efficiency evidence changed')
    comparison_config,provenance,comparison,selected,parent,manifest,rule=efficiency_choice(c['preparation'])
    if (sealed['comparison']!=comparison or sealed['comparison_provenance']!=provenance
            or sealed['selected_arm']!=selected or sealed['selection_rule']!=rule
            or sealed['parent_manifest_sha256']!=manifest or sealed['source_changes']!=list(FORMAL_SOURCES)
            or sealed['formal_launch_admitted'] is not False
            or sealed['formal_contract']!=formal_contract(t)
            or sealed['initial_policy_id']!=t['source_policy_id']):
        raise ValueError('formal package must bind freshly reconstructed efficiency selection')
    source=Path(parent['prepared'])/'package';original=read(source/'training.json')
    if (verify(source)!=manifest or normalized_training(original,parent['remote_root'])!=normalized_training(t,ops['remote_root'])
            or any(ops[k]!=parent[k] for k in ('host','lease','model','driver_root','shared_root'))):
        raise ValueError('formal training must preserve selected parent settings and leased devices')
    actual=source_inventory(package);prior=source_inventory(source)
    if (set(actual)!=set(prior)|set(FORMAL_SOURCES)
            or actual!=read(package/'RUNTIME-SOURCES.json')
            or t['dual_runtime_identity']['source_manifest_sha256']!=file_hash(package/'RUNTIME-SOURCES.json')
            or any(actual[n]!=h for n,h in prior.items())
            or any(actual[n]!=file_hash(Path(c['preparation']['project'])/n) for n in FORMAL_SOURCES)):
        raise ValueError('formal runtime may only add the declared formal entries')
    # Everything other than declared configuration/metadata and additions is
    # inherited byte-for-byte, including Stage1 policy, adapter and schedule.
    ignored={'FILES.json','FORMAL.json','RUNTIME-SOURCES.json','training.json','service-0.json','service-1.json',*FORMAL_SOURCES}
    old_files={str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file() and str(p.relative_to(source)) not in ignored}
    new_files={str(p.relative_to(package)):file_hash(p) for p in package.rglob('*') if p.is_file() and str(p.relative_to(package)) not in ignored}
    if old_files!=new_files:raise ValueError('formal policy, data or inherited assets differ')
    internal={ops['remote_root']+'/package/'+name:h for name,h in read(package/'FILES.json').items()
              if name not in ('training.json','service-0.json','service-1.json')}
    external={p:h for p,h in original['pins'].items() if not Path(p).is_relative_to(parent['remote_root']+'/package')}
    if t['pins']!={**external,**internal}:
        raise ValueError('complete formal dependency pins required')
    for i in range(2):
        expected=relocate(read(source/f'service-{i}.json'),parent['remote_root'],ops['remote_root'])
        observed=read(package/f'service-{i}.json')
        expected.update(pins=internal,dual_runtime_identity=t['dual_runtime_identity'])
        if expected!=observed:raise ValueError('formal service changed selected capacity or execution settings')
    schedule=audit_schedule(t,package)
    if sealed['schedule']!=schedule or schedule['real_exposures']!=81026 or schedule['updates']!=634:
        raise ValueError('full two-epoch source schedule required')
    tail_ops=c['tail_operations'];tail=audit_tail(tail_ops)
    tail_package=Path(tail_ops['prepared'])/'package';tail_t=read(tail_package/'training.json')
    if (tail['status']!='finite_dp2_tail_outputs_audited'
            or tail['files_sha256']!=tail_ops['manifest_sha256'] or tail['image']!=receipt['image']
            or tail['tail_admission']['real_queries']!=[65,128]
            or normalized_training(tail_t,tail_ops['remote_root'])!=normalized_training(t,ops['remote_root'])
            or any(tail_ops[k]!=ops[k] for k in ('host','lease','model','driver_root','shared_root'))):
        raise ValueError('actual same-objective tail NPU acceptance required')
    # Tail may have used the other service capacity arm; arithmetic and source
    # code must still match. Capacity itself is admitted by the completed100 pair.
    tail_sources=source_inventory(tail_package)
    if (any(tail_sources.get(n)!=h for n,h in prior.items())
            or tail_sources.get(FORMAL_SOURCES[0])!=actual[FORMAL_SOURCES[0]]):
        raise ValueError('tail diagnostic uses a different underlying runtime')
    return dict(status='fresh_formal_launch_admitted',formal_manifest=ops['manifest_sha256'],
        selected_arm=selected,selection_rule=rule,source_parent=parent,schedule=schedule,
        tail_audit=tail,sparse_npu_recovery_verified=False,automatic_resume_allowed=False,
        source_evaluation_steps=t['evaluation_steps'],target_tuning_allowed=False)


def formal_contract(t):
    from tools.reports.context_macro_formal_candidate import contract
    return contract(t)


def run(c,provenance):
    if c.get('formal_training_dispatch_allowed') is not True:
        raise ValueError('explicit formal dispatch configuration required')
    proof=admit(c)
    ops=dict(c['operations'],finite_inference_dispatch_allowed=True)
    # The lifecycle records this freshly computed proof in attempt.json, takes
    # the shared lease, checks actual idle devices and retains exact handles.
    return run_services(ops,dict(provenance,formal_launch_admission=proof),
        validator=formal.validate,command_builder=formal.command,required_devices=(4,5,6,7))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    parser.add_argument('--action',choices=['admit','run'],default='admit')
    args=parser.parse_args();c,p=load_config(args.config,args.local)
    import torch
    torch.set_num_threads(1)
    print(json.dumps(run(c,p) if args.action=='run' else admit(c),indent=2))

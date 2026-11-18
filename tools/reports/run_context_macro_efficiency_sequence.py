"""Finish the declared micro8/16/32 study without polling or model calls.

Wait on the existing finalizer, audit it, then run/collect each fresh successor
once using the shared device lease. Failure preserves handles and stops.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.launch_context_batch_probe_candidate import write_new
from tools.reports.finalize_context_macro_training import wait_driver
from tools.reports.context_macro_training_operations import validate,command,run as train
from tools.reports.audit_context_macro_training import audit_collected,audit_outputs
from tools.reports.audit_context_macro_probe import collect_and_audit
from tools.reports.context_macro_efficiency_summary import summarize
from tools.reports.compare_context_macro_efficiency import compare,controls


def read(path):return json.loads(Path(path).read_text())


def validate_sequence(config):
    if not {config['comparison_config'],config['comparison_local'],config['predecessor_finalizer']}<=set(config['pins']):
        raise ValueError('pinned comparison and predecessor files required')
    for name,digest in config['pins'].items():
        if file_hash(name)!=digest:raise ValueError('sequence dependency changed: '+name)
    comparison,provenance=load_config(config['comparison_config'],config['comparison_local'])
    if list(comparison['arms'])!=['micro8','micro16','micro32'] or comparison['warmup_updates']!=2:
        raise ValueError('fixed ordered micro8/16/32 study required')
    if file_hash(comparison['plan'])!=comparison['plan_sha256']:raise ValueError('efficiency plan changed')
    plan=read(comparison['plan'])
    if plan['steps_per_arm']!=100:raise ValueError('exact 100-update study required')
    operations={};common=None
    for name,spec in comparison['arms'].items():
        if not {spec['config'],spec['local']}<=set(config['pins']):
            raise ValueError('pinned per-arm configuration required')
        ops,p=load_config(spec['config'],spec['local']);receipt=validate(ops)
        expected=plan['arms'][name];package=Path(ops['prepared'])/'package';t=read(package/'training.json')
        if (receipt['status']!='macro_efficiency_prepared_only' or ops['manifest_sha256']!=expected['manifest_sha256']
                or ops['remote_root']!=expected['remote_root'] or t['microbatch']!=expected['microbatch']
                or t['stop_update']!=100 or t['start_update']!=0 or ops['finite_training_dispatch_allowed'] is not True):
            raise ValueError('declared fresh efficiency arm required')
        matched=(controls(t,[read(package/f'service-{i}.json') for i in range(2)],ops['remote_root']),
                 ops['host'],ops['lease'],file_hash(package/'RUNTIME-SOURCES.json'))
        if common is None:common=matched
        elif common!=matched:raise ValueError('efficiency configurations differ beyond microbatch')
        if name!='micro8' and (Path(ops['operations_root'])/'attempt.json').exists():
            raise ValueError('successor already attempted; preserve exact handles')
        operations[name]=(ops,p)
    baseline=Path(operations['micro8'][0]['operations_root'])
    if config['predecessor_finalizer']!=str(baseline/'finalization-driver.json'):
        raise ValueError('wait only for baseline finalizer')
    return operations,comparison,provenance


def run(config,provenance):
    arms,comparison,_=validate_sequence(config)
    root=Path(config['output']);check_paths({'efficiency_sequence':root})
    root.mkdir(parents=True,exist_ok=False)
    write_new(root/'config.json',config);write_new(root/'provenance.json',provenance)
    try:
        baseline=Path(arms['micro8'][0]['operations_root'])
        predecessor=read(config['predecessor_finalizer'])
        write_new(root/'waiting.json',dict(process=predecessor,terminal=str(baseline/'finalization-terminal.json')))
        wait_driver(predecessor,baseline/'finalization-terminal.json')
        # Revalidate after waiting; source edits cannot silently change successors.
        arms,comparison,_=validate_sequence(config)
        terminal=read(baseline/'finalization-terminal.json')
        if (terminal['status']!='finite_training_collected_audited'
                or terminal['audit_status']!='finite_dp2_efficiency_audited'):
            raise ValueError('baseline finalization did not pass; preserve attempt')
        initial=audit_collected(arms['micro8'][0])
        if initial['status']!='finite_dp2_efficiency_audited':raise ValueError('baseline re-audit failed')
        write_new(root/'micro8-audit.json',initial)
        for name in ('micro16','micro32'):
            for path,digest in config['pins'].items():
                if file_hash(path)!=digest:raise ValueError('sequence dependency changed: '+path)
            ops,pp=arms[name]
            write_new(root/(name+'-dispatch.json'),dict(operations=ops))
            train(ops,pp)  # Existing exact-container waits, local storage and device lease.
            result=collect_and_audit(ops,validator=validate,command_builder=command,
                output_auditor=audit_outputs,result_dirs=('run','journal','service-0','service-1'))
            if result['status']!='finite_dp2_efficiency_audited':raise ValueError('successor audit failed')
            base=Path(ops['operations_root'])
            timing=summarize(base/'collected/run',start_update=0,stop_update=100,warmup_updates=2)
            write_new(base/'efficiency-summary.json',timing)
            write_new(base/'finalization-terminal.json',dict(status='finite_training_collected_audited',
                audit_status=result['status'],owner='efficiency_sequence',restart=False))
            write_new(root/(name+'-audit.json'),result)
        result=compare(comparison)
        write_new(comparison['output'],result)
        write_new(root/'terminal.json',dict(exit_code=0,status='three_arm_efficiency_comparison_completed',
            comparison=comparison['output'],formal_training_dispatched=False))
        return result
    except BaseException as exc:
        write_new(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    parser.add_argument('--action',choices=['validate','run'],default='validate')
    args=parser.parse_args();config,provenance=load_config(args.config,args.local)
    import torch
    torch.set_num_threads(1)
    if args.action=='run':run(config,provenance)
    else:
        validate_sequence(config);print('validated efficiency sequence; no dispatch')

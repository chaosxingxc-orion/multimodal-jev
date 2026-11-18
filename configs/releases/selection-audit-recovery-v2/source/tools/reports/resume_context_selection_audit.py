"""Explicit recovery of a terminal selection-audit failure, without retraining.

Existing published packages, original failure logs and completed inference are
immutable. Only absent audits and subsequent planned points may be produced.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from multimodal_jev.local_config import load_config
from tools.reports.committed_signal_report import check_paths, read, write
from tools.jev.collect_context_rl_training import sha
from tools.reports.run_context_arithmetic_experiment import plan
from tools.jev.prepare_context_formal_package import PREFIX


def validate(config):
    keys={'schema','experiment_root','failed_status_sha256','accepted_audit',
          'accepted_audit_sha256','resume_step','output_directory'}
    if config.get('schema')=='selection-audit-recovery-v2':
        keys|={'relocation_evidence','relocation_intent_sha256','pins'}
    if (set(config) != keys
            or config['schema'] not in ('selection-audit-recovery-v1','selection-audit-recovery-v2')):
        raise ValueError('Explicit selection recovery configuration required')
    root=Path(config['experiment_root']).resolve()
    if sha(root/'status.json') != config['failed_status_sha256']:
        raise ValueError('Original stopped status changed')
    stopped=read(root/'status.json')
    if (stopped['phase']!='stopped_preserve_handles' or stopped.get('restart') is not False
            or 'audit_context_rl_selection_v2' not in stopped.get('error','')):
        raise ValueError('Exact audit failure required; no training recovery')
    spec=read(root/'input.json');planned=plan(spec)
    if planned!=read(root/'plan.json'):raise ValueError('Frozen experiment plan changed')
    step=config['resume_step'];point=root/f'step-{step:04d}'
    if type(step) is not int or step not in planned['evaluation_steps'] or (point/'audit.json').exists():
        raise ValueError('Absent audit for a planned existing point required')
    accepted=Path(config['accepted_audit'])
    if sha(accepted)!=config['accepted_audit_sha256']:raise ValueError('Acceptance proof changed')
    proof=read(accepted);transport=read(point/'collected.transport.json')
    if (proof['passed'] is not True or proof['step']!=step or proof['run_id']!=planned['run_id']
            or proof['algorithm']!=planned['algorithm'] or proof['files']!=transport['files']
            or proof['container_id']!=transport['container_id']
            or proof['state']['Status']!='exited' or proof['state']['ExitCode']!=0
            or proof['state']['OOMKilled']):
        raise ValueError('Accepted exact terminal collection required')
    for name,digest in transport['files'].items():
        if sha(point/'collected'/name)!=digest:raise ValueError('Accepted collection changed')
    index=planned['evaluation_steps'].index(step)
    for prior in planned['evaluation_steps'][:index]:
        if read(root/f'step-{prior:04d}/audit.json')['passed'] is not True:
            raise ValueError('Earlier audit missing')
    for future in planned['evaluation_steps'][index+1:]:
        tag='context-rl-eval-'+planned['run_id'].removeprefix('context-rl-')+f'-step{future:04d}'
        if (root/f'step-{future:04d}').exists() or any(
                (root.parent/(tag+suffix)).exists() for suffix in ('-launch.json','-attempt.json')):
            raise ValueError('Later work already exists; reconcile exact handles')
    return root,spec,planned


def relocation_complete(config, root):
    evidence=Path(config['relocation_evidence'])
    if sha(evidence/'intent.json')!=config['relocation_intent_sha256']:
        raise ValueError('Relocation intent changed')
    intent=read(evidence/'intent.json')['config']
    completed=read(evidence/'COMPLETE.json')
    collection=root/'training-collected'
    if (intent['collection']!=str(collection) or intent['owner_lock']!=str(root/'owner.lock')
            or completed['status']!='relocated_verified_original_paths_preserved'
            or completed['source']!=str(collection) or completed['destination']!=intent['destination']
            or collection.resolve()!=Path(intent['destination'])/'collection'
            or sha(intent['transport'])!=intent['transport_sha256']
            or sha(intent['proof'])!=intent['proof_sha256']):
        raise ValueError('Exact completed original collection relocation required')


def verify_recovery_pins(config):
    if config.get('schema')!='selection-audit-recovery-v2':return
    pins=config['pins']
    if not isinstance(pins,dict) or not pins:raise ValueError('Recovery source pins required')
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Recovery dependency changed: '+path)


def run(template,local):
    config,provenance=load_config(template,local)
    verify_recovery_pins(config)
    output=Path(config['output_directory']).resolve()
    root=Path(config['experiment_root']).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Separate recovery evidence directory required')
    # Original owner.lock must already exist; never recreate a missing owner.
    with (root/'owner.lock').open('r') as owner:
        after_relocation=config.get('schema')=='selection-audit-recovery-v2'
        # Wait on the exact migration lease, no polling or model scheduling.
        fcntl.flock(owner,fcntl.LOCK_EX|(0 if after_relocation else fcntl.LOCK_NB))
        if after_relocation:relocation_complete(config,root)
        verify_recovery_pins(config)
        check_paths({'recovery':output,'experiment':root})
        root,spec,p=validate(config)
        output.mkdir(exist_ok=False)
        write(output/'provenance.json',provenance)
        write(output/'original-stopped-status.json',read(root/'status.json'))
        write(output/'input.json',config)
        def status(phase,**extra):
            write(output/'status.json',dict(phase=phase,**extra))
        def invoke(module,arguments,name):
            verify_recovery_pins(config)
            status('executing',operation=name)
            argv=[sys.executable,'-m']
            if module in ('audit_context_rl_selection_v2','audit_context_rl_curve_v2'):
                argv+=['tools.reports.context_metric_python312','--module','tools.jev.'+module,'--']
            else:argv+=['tools.jev.'+module]
            argv+=list(map(str,arguments))
            with (output/(name+'.log')).open('x') as log:
                subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True)
        common=dict(experiment=spec['experiment'],stage1_root=spec['stage1_root'],stage1_proof=spec['stage1_proof'],
                    rl_root=str(root/'training-collected'),rl_proof=str(root/'training-audit.json'),
                    run_id=p['run_id'],algorithm=p['algorithm'])
        points=[]
        try:
            for step in p['evaluation_steps']:
                stem=f'step-{step:04d}';point=root/stem
                if step<=config['resume_step']:
                    entry=read(point/'audit-spec.json')
                    if step==config['resume_step']:
                        invoke('audit_context_rl_selection_v2',['--spec',point/'audit-spec.json',
                               '--output',point/'audit.json'],stem+'-audit')
                        if read(point/'audit.json')!=read(config['accepted_audit']):
                            # NLL/Brier libm/set iteration rounding is already
                            # bounded by the original comparer; preserve each
                            # audit's diagnostic detail but require same result.
                            a,b=read(point/'audit.json'),read(config['accepted_audit'])
                            a.pop('metric_comparison');b.pop('metric_comparison')
                            if a!=b:raise ValueError('Repeated accepted audit differs')
                else:
                    point.mkdir(exist_ok=False)
                    write(point/'prepare-spec.json',dict(common,template=spec['template'],step=step))
                    package=point/'package'
                    remote=PREFIX+'context-rl-eval-'+p['run_id'].removeprefix('context-rl-')+f'-step{step:04d}'
                    invoke('prepare_context_rl_evaluation_package',['--spec',point/'prepare-spec.json',
                           '--output',package,'--remote-root',remote],stem+'-prepare')
                    invoke('launch_context_rl_evaluation_v2',['--package',package,'--manifest-sha256',
                           sha(package/'PACKAGE.json'),'--launch'],stem+'-launch')
                    launch_path=root.parent/(Path(remote).name+'-launch.json')
                    collection=point/'collected'
                    invoke('collect_context_rl_selection',['--launch',launch_path,'--output',collection],stem+'-collect')
                    entry=dict(common,step=step,collection=str(collection),transport=str(collection.with_suffix('.transport.json')),
                               expected_config_sha256=read(launch_path)['config_sha256'])
                    write(point/'audit-spec.json',entry)
                    invoke('audit_context_rl_selection_v2',['--spec',point/'audit-spec.json',
                           '--output',point/'audit.json'],stem+'-audit')
                points.append({k:entry[k] for k in ('step','collection','transport','expected_config_sha256')})
            if (root/'curve-spec.json').exists() or (root/'curve.json').exists():
                raise ValueError('Unexpected prior curve output')
            write(root/'curve-spec.json',dict(common,evaluations=points))
            invoke('audit_context_rl_curve_v2',['--spec',root/'curve-spec.json','--output',root/'curve.json'],'audit-curve')
            if read(root/'curve.json')['complete'] is not True:raise ValueError('Incomplete recovered curve')
            write(root/'status.json',dict(phase='complete_training_and_selection_curve',run_id=p['run_id'],
                points=len(points),recovery=str(output),scope='One complete curve; transfer and comparison separate'))
            status('complete',curve_sha256=sha(root/'curve.json'))
        except BaseException as exc:
            status('stopped_preserve_handles',error=repr(exc),restart=False)
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--local',default=os.environ.get('JEV_LOCAL_CONFIG','.local/config/macros.json'))
    args=parser.parse_args();run(args.config,args.local)

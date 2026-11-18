"""Versioned Stage2 acceptance controller with parallel Stage1 provenance.

Retire only the exact CPU-only predecessor at an idle four-card boundary.
Preserve all frozen packages; run the same objective prefix/recovery sequence.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports.prepare_objective_training_candidate import read_bound
from tools.reports.context_async_pipeline import exact_process
from tools.reports.context_isolated_source_audit import IsolatedSourceAuditEngine
from tools.reports.objective_parallel_admission import implementation
from tools.reports import run_objective_acceptance as original


def retire(c,root):
    record=json.loads(read_bound(c['previous_driver']));prior=json.loads(read_bound(c['previous_config']))
    if 'tools.reports.run_objective_acceptance' not in record['argv']:
        raise ValueError('exact original objective controller required')
    if (prior['prefix']['lease']!=c['sequence']['prefix']['lease']
            or prior['prefix']['host']!=c['sequence']['prefix']['host']):
        raise ValueError('same four-card ownership required')
    fd=os.pidfd_open(record['pid']);stopped=False
    try:
        exact_process(record);signal.pidfd_send_signal(fd,signal.SIGSTOP);stopped=True
        deadline=time.monotonic()+2
        while exact_process(record) not in ('T','t'):
            if time.monotonic()>deadline:raise ValueError('controller stop failed')
            time.sleep(.01)
        for task in Path(f"/proc/{record['pid']}/task").glob('*/children'):
            if task.read_text().strip():raise ValueError('active child work; reconcile exact handles')
        for name in ['prefix','recovery']:
            op=Path(prior[name]['operations_root'])
            if list(op.glob('*-attempt.json')) or list(op.glob('*-launch.json')):
                raise ValueError('NPU dispatch already attempted; retain original controller')
        with Path(prior['prefix']['lease']).open('r') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            usage=subprocess.check_output(['ssh','-o','ConnectTimeout=10',prior['prefix']['host'],'npu-smi info'],text=True,timeout=55)
            if any(f'No running processes found in NPU {d}' not in usage for d in c['devices']):
                raise ValueError('four-card device busy')
            commit_json(root/'handoff.json',dict(previous=record,reason='same_validators_parallel_source_replay',
                remote_processes_signalled=False,preserve_frozen_packages=True))
            signal.pidfd_send_signal(fd,signal.SIGTERM);signal.pidfd_send_signal(fd,signal.SIGCONT);stopped=False
            poll=select.poll();poll.register(fd,select.POLLIN)
            if not poll.poll(10000):raise ValueError('old CPU controller failed to exit')
    finally:
        if stopped:
            try:signal.pidfd_send_signal(fd,signal.SIGCONT)
            except ProcessLookupError:pass
        os.close(fd)


def run(c,provenance):
    for pin in c['pins']:read_bound(pin)
    original.validate(c['sequence'])
    root=Path(c['output']);root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    try:
        retire(c,root)
        with IsolatedSourceAuditEngine(cpu_workers=c['cpu_workers'],io_workers=c['io_workers']) as engine:
            result=implementation(engine).run(c['sequence'],provenance)
        commit_json(root/'terminal.json',result);return result
    except BaseException as e:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(e),restart=False,preserve_handles=True));raise


def detach(template,local):
    c,_=load_config(template,local);original.validate(c['sequence'])
    root=Path(c['output']).with_name(Path(c['output']).name+'-driver');root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m','tools.reports.run_stage2_parallel_acceptance','--config',str(Path(template).resolve()),
        '--local',str(Path(local).resolve()),'--action','run']
    with (root/'driver.log').open('x') as log:
        p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=p.pid,argv=argv,started_unix=time.time(),process_start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split(') ',1)[1].split()[19])
    commit_json(root/'started.json',record);return record


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['run','detach'],default='run');a=p.parse_args();c,prov=load_config(a.config,a.local)
    print(json.dumps(detach(a.config,a.local) if a.action=='detach' else run(c,prov),indent=2))

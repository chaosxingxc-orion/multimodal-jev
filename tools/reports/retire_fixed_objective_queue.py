"""Explicit queue handoff at an idle device boundary; preserve all artifacts.

Only named local controllers and their verified CPU audit children are retired.
Remote containers are inspected, never signalled. No retries or polling service.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports.context_async_pipeline import exact_process
from tools.reports.prepare_objective_training_candidate import read_bound


def run(c, provenance):
    if c.get('replace_fixed_queue') is not True:
        raise ValueError('explicit queue replacement required')
    records=[json.loads(read_bound(x)) for x in c['drivers']]
    if len(records)!=2 or [r['argv'][3] for r in records]!=[
            'tools.reports.objective_acceptance_successor',
            'tools.reports.context_stage1_priority_pipeline']:
        raise ValueError('exact waiting successor then original controller required')
    prior=json.loads(read_bound(c['prior_config']))
    if prior['deployment']['lease']!=c['lease'] or prior['deployment']['host']!=c['host']:
        raise ValueError('same resource ownership required')
    root=Path(c['output']);root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    handles=[];children=[];committed=False
    try:
        for r in records:
            fd=os.pidfd_open(r['pid']);handles.append((fd,r));exact_process(r)
            signal.pidfd_send_signal(fd,signal.SIGSTOP)
            deadline=time.monotonic()+2
            while exact_process(r) not in ('T','t'):
                if time.monotonic()>deadline:raise ValueError('controller stop failed')
                time.sleep(.01)
        # Recursively retain an exact identity for CPU-only audit descendants.
        pending=[r['pid'] for r in records]
        while pending:
            parent=pending.pop()
            ids=set()
            for task in Path(f'/proc/{parent}/task').glob('*/children'):
                ids.update(int(v) for v in task.read_text().split())
            for pid in sorted(ids):
                proc=Path(f'/proc/{pid}')
                argv=proc.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
                if not (len(argv)>=4 and argv[1:3]==['-B','-c'] and
                        argv[3].startswith(('from multiprocessing.resource_tracker import main;',
                                            'from multiprocessing.spawn import spawn_main;'))):
                    raise ValueError('active non-audit child; reconcile instead of preempting: '+str(pid))
                r=dict(pid=pid,argv=argv,process_start_ticks=proc.joinpath('stat').read_text().split(') ',1)[1].split()[19])
                fd=os.pidfd_open(pid);exact_process(r);signal.pidfd_send_signal(fd,signal.SIGSTOP)
                children.append((fd,r));pending.append(pid)
        with Path(c['lease']).open('r') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            def remote(args):
                import shlex
                return subprocess.check_output(['ssh','-o','ConnectTimeout=10',c['host'],shlex.join(args)],text=True,timeout=55)
            usage=remote(['npu-smi','info'])
            if any(f'No running processes found in NPU {d}' not in usage for d in c['devices']):
                raise ValueError('leased NPU busy; preserve original queue')
            inspections=[]
            for p in sorted(Path(prior['operations_root']).rglob('*-launch.json')):
                launch=json.loads(p.read_text());state=json.loads(remote(['docker','inspect',launch['container_id']]))[0]
                if state['Id']!=launch['container_id'] or state['State']['Running']:
                    raise ValueError('remote owned job is active')
                inspections.append(dict(launch=str(p),container_id=state['Id'],state=state['State']))
            for p in Path(prior['operations_root']).glob('*/*-attempt.json'):
                if not p.with_name(p.name.replace('-attempt.json','-launch.json')).exists():
                    raise ValueError('unreconciled attempted dispatch')
            commit_json(root/'intent.json',dict(reason='user_requested_editable_objective_for_next_round',
                records=records,children=[r for _,r in children],inspections=inspections,
                obsolete_pending_stages=['fixed_tail','fixed_recovery','fixed_efficiency','fixed_formal'],
                preserve_all_artifacts=True,remote_processes_signalled=False))
            committed=True
            for fd,_ in handles+children:
                signal.pidfd_send_signal(fd,signal.SIGTERM);signal.pidfd_send_signal(fd,signal.SIGCONT)
            for fd,_ in handles+children:
                poll=select.poll();poll.register(fd,select.POLLIN)
                if not poll.poll(10000):raise ValueError('local process did not exit')
            result=dict(status='fixed_queue_retired_at_idle_boundary',exit_code=0,
                next_entrypoint='tools.reports.run_objective_acceptance',
                remote_processes_signalled=False,old_artifacts_preserved=True)
            commit_json(root/'terminal.json',result);return result
    except BaseException as e:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(e),committed=committed));raise
    finally:
        for fd,_ in handles+children:
            if not committed:
                try:signal.pidfd_send_signal(fd,signal.SIGCONT)
                except ProcessLookupError:pass
            os.close(fd)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local);print(json.dumps(run(c,provenance),indent=2))

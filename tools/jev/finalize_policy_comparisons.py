#!/usr/bin/env python3
"""Observe existing evaluation containers and compare only completed predictions.

Never creates/restarts training or inference. Observation failures are retried;
only an inspected exit-zero container plus valid receipts enables scoring.
"""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.compare_policies import compare


class ObservationUnavailable(RuntimeError):
    pass


def inspect_container(host,name):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',host) or host.startswith('-'):
        raise ValueError('explicit SSH host alias required')
    command=shlex.join(['docker','inspect','--format',
        '{"id":{{json .Id}},"state":{{json .State}}}',name])
    try:
        proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,command],
                            capture_output=True,text=True,timeout=20)
    except subprocess.TimeoutExpired as exc:
        raise ObservationUnavailable('Docker observation timed out; no terminal inference') from exc
    if proc.returncode:
        raise ObservationUnavailable(f'Docker observation failed ({proc.returncode}): {proc.stderr.strip()[:400]}')
    try:
        result=json.loads(proc.stdout)
        if not result['id'] or not isinstance(result['state'],dict):raise ValueError('invalid state')
        return result
    except (KeyError,ValueError) as exc:
        raise ObservationUnavailable('Docker returned no usable state') from exc


def finalize_job(job,output,*,inspector=inspect_container):
    """One polling pass; returns waiting, observation_retry, or completed."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    states={}
    for role in ('base','candidate'):
        dep=job[role]
        try:states[role]=inspector(dep['host'],dep['container'])
        except ObservationUnavailable as exc:
            return {'phase':'observation_retry','role':role,'error':str(exc)}
        if states[role]['id']!=dep['container_id']:
            raise ValueError(f'{role} container identity changed; inspect without restarting')
        state=states[role]['state']
        if state['Status']=='exited' and state['ExitCode']!=0:
            raise RuntimeError(f'{role} evaluation exited {state["ExitCode"]}')
        if state['Status'] in {'dead','removing'}:
            raise RuntimeError(f'{role} evaluation is {state["Status"]}')
    if any(x['state']['Status']!='exited' or x['state']['Running'] for x in states.values()):
        return {'phase':'waiting_for_evaluation','containers':states}
    manifest=Path(job['manifest'])
    if file_hash(manifest)!=job['manifest_sha256']:raise ValueError('frozen manifest changed')
    # compare checks complete prediction IDs, terminal receipts, raw artifact
    # hashes and equal decoding protocols, then uses one current scorer.
    report=compare(manifest,Path(job['base']['run']),Path(job['candidate']['run']),
                   draws=job.get('draws',2000),seed=job.get('seed',17))
    report['observed_containers']=states
    path=output/'comparison.json'
    if path.exists():
        if json.loads(path.read_text())!=report:raise ValueError('completed comparison changed')
    else:atomic_json(path,report)
    for role in ('base','candidate'):
        metrics=output/(role+'-common-metrics.json')
        if metrics.exists():
            if json.loads(metrics.read_text())!=report['rescored_'+role]:raise ValueError('saved common metrics changed')
        else:atomic_json(metrics,report['rescored_'+role])
    return {'phase':'completed','comparison':str(path),'comparison_sha256':file_hash(path),
            'base_all':report['rescored_base']['by_variant']['all'],
            'candidate_all':report['rescored_candidate']['by_variant']['all']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'finalizer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.output/'terminal.json').exists():raise FileExistsError('terminal finalizer; inspect before retry')
        try:
            config=json.loads(args.config.read_text());source=Path(__file__).parents[2]
            identity={'config':config,'config_sha256':file_hash(args.config),
                      'source_sha256':{str(p.relative_to(source)):file_hash(p) for p in
                          [Path(__file__),*sorted((source/'src/multimodal_jev').glob('*.py'))]}}
            target=args.output/'identity.json'
            if target.exists() and json.loads(target.read_text())!=identity:raise ValueError('finalizer identity changed')
            atomic_json(target,identity)
            states={}
            while True:
                for name,job in config['jobs'].items():
                    if not re.fullmatch(r'[a-z0-9_-]+',name):raise ValueError('invalid job name')
                    if states.get(name,{}).get('phase')=='completed':continue
                    states[name]=finalize_job(job,args.output/name)
                atomic_json(args.output/'progress.json',{'jobs':states,'heartbeat_unix':time.time()})
                if all(s['phase']=='completed' for s in states.values()):break
                time.sleep(30)
            atomic_json(args.output/'result.json',{'jobs':states,
                'scope':'completed paired development comparisons; no RL or final generalization claim'})
            atomic_json(args.output/'terminal.json',{'exit_code':0,'finished_unix':time.time()})
        except BaseException as exc:
            atomic_json(args.output/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()})
            raise


if __name__=='__main__':main()

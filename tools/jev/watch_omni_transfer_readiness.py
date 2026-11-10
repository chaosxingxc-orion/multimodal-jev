#!/usr/bin/env python3
"""Observe exact dependencies and prepare, but never dispatch, full transfer."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

from multimodal_jev.baseline import atomic_json,file_hash
from watch_omni_selection_to_test import process_start
from watch_qwen3_omni_probe import inspect as inspect_container
from register_omni_slurp_transfer import register
from run_omni_slurp_transfer_queue import prepare
from verify_omni_transfer_train import verify


def observe_dependency(spec):
    root=Path(spec['root'])
    for name,digest in spec['files'].items():
        if file_hash(root/name)!=digest:raise ValueError('dependency identity changed: '+name)
    terminal=root/'evaluation-terminal.json'
    if not terminal.exists():
        if process_start(spec['pid'])!=spec['process_start']:
            raise RuntimeError('dependency supervisor missing or replaced without terminal')
        return False
    if json.loads(terminal.read_text())['exit_code']!=0:
        raise RuntimeError('dependency failed; transfer remains undispatched')
    for child in spec['container_children']:
        launch=json.loads((root/child/'launch.json').read_text())
        item=inspect_container(launch['container_id'])
        if (item['Id']!=launch['container_id'] or item['Image']!=launch['image']
                or item['State']['Status']!='exited' or item['State']['ExitCode']!=0):
            raise ValueError('exact successful dependency container required')
    return True


def estimate(validation_root,queue):
    """Account observed runtime, never read targets or select on task quality."""
    identity=json.loads((validation_root/'run/identity.json').read_text())
    result=json.loads((validation_root/'run/result.json').read_text())
    predictions=[json.loads(line) for line in (validation_root/'run/predictions.jsonl').open()]
    if len(predictions)!=336 or result['requests']!=336:raise ValueError('complete TRAIN cost sample required')
    seconds=[p['seconds'] for p in predictions]
    if any(type(x) not in (int,float) or not 0<=x<float('inf') for x in seconds):
        raise ValueError('finite nonnegative measured inference time required')
    queue_intent=json.loads((queue/'queue-intent.json').read_text())
    prompts={r['prompt'] for r in queue_intent['roster']}
    request_count=queue_intent['requests_per_prompt'];mean=sum(seconds)/len(seconds)
    elapsed=result['elapsed_seconds']
    if type(elapsed) not in (int,float) or not sum(seconds)<=elapsed<float('inf'):
        raise ValueError('finite total runtime must cover measured generations')
    overhead=elapsed-sum(seconds)
    generations=mean*request_count*len(prompts)
    estimated=generations+overhead*len(queue_intent['roster'])
    return dict(schema='jev-transfer-runtime-estimate-v1',train_result_sha256=file_hash(validation_root/'run/result.json'),
        observed_requests=len(seconds),observed_recordings=2,observed_source_groups=1,
        max_new_tokens=identity['config']['max_new_tokens'],mean_generation_seconds=mean,
        observed_engine_and_hash_overhead_seconds=overhead,requests_per_prompt=request_count,prompts=len(prompts),
        shard_container_count=len(queue_intent['roster']),estimated_generation_days=generations/86400,
        estimated_serial_wall_days=estimated/86400,estimated_npu_device_hours=estimated*2/3600,
        estimate_is_not_budget_or_completion=True,
        limitations='One TRAIN group and parent prompt only; candidate, audio length, task mix and shared-host load may change throughput. Excludes external scoring, queue observation delays and Docker startup. No quality-based sampling or schedule reduction.',
        action='Full queue prepared only; no inference dispatched by readiness observer')


def execute(root):
    intent=json.loads((root/'intent.json').read_text())
    if intent['dispatch'] is not False:raise ValueError('readiness observer is prepare-only')
    if set(intent['dependencies'])!={'selection','train_validation'}:
        raise ValueError('both selection and TRAIN validation dependencies required')
    if type(intent['observation_seconds']) is not int or not 1<=intent['observation_seconds']<=100000:
        raise ValueError('bounded dependency observation required')
    code=Path(intent['coordinator_source'])
    for relative,digest in intent['source_files'].items():
        if file_hash(code/relative)!=digest:raise ValueError('coordinator source changed')
    if file_hash(intent['shard_plan'])!=intent['shard_plan_sha256']:raise ValueError('full shard plan changed')
    deadline=time.monotonic()+intent['observation_seconds']
    while True:
        ready={}
        try:
            for key,spec in intent['dependencies'].items():ready[key]=observe_dependency(spec)
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,json.JSONDecodeError) as exc:
            atomic_json(root/'progress.json',dict(phase='observation_retry',error_type=type(exc).__name__,heartbeat_unix=time.time()))
        else:
            atomic_json(root/'progress.json',dict(phase='waiting_dependencies',ready=ready,heartbeat_unix=time.time()))
            if all(ready.values()):break
        if time.monotonic()>deadline:raise TimeoutError('observation expired; no restart or TEST dispatch')
        time.sleep(20)
    with (root/'preparation-intent.json').open('x') as handle:
        json.dump(dict(started_unix=time.time(),dispatch=False,automatic_retry=False),handle)
    selection=Path(intent['dependencies']['selection']['root'])
    validation=Path(intent['dependencies']['train_validation']['root'])
    protocol=root/'transfer-protocol.json';queue=Path(intent['queue_root'])
    verify(validation)
    register(selection,Path(intent['shard_plan']),protocol)
    prepared=prepare(queue,protocol,validation,Path(intent['native_probe']),intent['physical_devices'])
    atomic_json(root/'prepared-queue.json',prepared)
    atomic_json(root/'runtime-estimate.json',estimate(validation,queue))
    atomic_json(root/'progress.json',dict(phase='prepared_not_dispatched',heartbeat_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    with (a.root/'observer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.root/'terminal.json').exists() or (a.root/'preparation-intent.json').exists():
            raise FileExistsError('existing readiness attempt; no automatic replay')
        try:execute(a.root)
        except BaseException as exc:
            atomic_json(a.root/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time(),dispatched=False))
            raise
        atomic_json(a.root/'terminal.json',dict(exit_code=0,finished_unix=time.time(),dispatched=False))

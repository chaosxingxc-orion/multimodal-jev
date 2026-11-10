#!/usr/bin/env python3
"""One-shot complete transfer queue, gated by real TRAIN engine validation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.slurp_jev_shards import plan_shards
from prepare_omni_slurp_transfer_shard import prepare as prepare_shard
from watch_qwen3_omni_probe import run as dispatch
from verify_omni_transfer_train import verify


def prepare(root,protocol_path,validation_root,native_probe,devices):
    validation=verify(validation_root)
    protocol=json.loads(protocol_path.read_text());plan_path=Path(protocol['shard_plan'])
    if file_hash(plan_path)!=protocol['shard_plan_sha256']:raise ValueError('shard plan changed')
    plan=json.loads(plan_path.read_text())
    if plan_shards(plan['release'],'test',plan['recordings_per_shard_target'])!=plan:
        raise ValueError('complete plan must reproduce')
    if (protocol['runtime']!=validation['runtime'] or protocol['image']!=validation['image']
            or protocol['versions']!=validation['versions'] or protocol['model_files']!=validation['model_files']
            or protocol['asset_audit_sha256']!=validation['asset_audit_sha256']
            or devices not in ([4,5],[6,7])):
        raise ValueError('validated same model and runtime required')
    root.mkdir(parents=True,exist_ok=False)
    (root/'transfer-protocol.json').write_bytes(protocol_path.read_bytes())
    atomic_json(root/'train-validation.json',validation)
    prompts=['parent',*sorted(set(protocol['prompts'])-{'parent'})]
    roster=[dict(shard=s['id'],prompt=p) for s in plan['shards'] for p in prompts]
    intent=dict(source_commit=validation['source_commit'],validation_root=str(validation_root.resolve()),
        validation_sha256=file_hash(root/'train-validation.json'),protocol_sha256=file_hash(protocol_path),
        native_probe=str(native_probe.resolve()),physical_devices=devices,roster=roster,
        requests_per_prompt=plan['decisions'],automatic_retry=False,dispatched=False)
    atomic_json(root/'queue-intent.json',intent)
    return intent


def run(root):
    intent=json.loads((root/'queue-intent.json').read_text());protocol_path=root/'transfer-protocol.json'
    if (file_hash(protocol_path)!=intent['protocol_sha256']
            or file_hash(root/'train-validation.json')!=intent['validation_sha256']
            or verify(Path(intent['validation_root']))!=json.loads((root/'train-validation.json').read_text())):
        raise ValueError('frozen queue evidence changed')
    protocol=json.loads(protocol_path.read_text());plan=json.loads(Path(protocol['shard_plan']).read_text())
    expected=[dict(shard=s['id'],prompt=p) for s in plan['shards']
              for p in ['parent',*sorted(set(protocol['prompts'])-{'parent'})]]
    if intent['roster']!=expected:raise ValueError('whole registered transfer roster required')
    for cell in expected:
        if (root/cell['prompt']/cell['shard']).exists():
            raise FileExistsError('prior preparation or dispatch; audit without automatic replay')
    with (root/'dispatch-intent.json').open('x') as handle:
        json.dump(dict(started_unix=time.time(),automatic_retry=False),handle)
    runs={p:{} for p in protocol['prompts']}
    for index,cell in enumerate(expected):
        child=root/cell['prompt']/cell['shard']
        atomic_json(root/'progress.json',dict(phase='prepare',done=index,total=len(expected),**cell,heartbeat_unix=time.time()))
        prepare_shard(child,protocol_path,cell['shard'],cell['prompt'],Path(intent['native_probe']),
                      intent['source_commit'],intent['physical_devices'])
        try:dispatch(child)
        except BaseException as exc:
            atomic_json(child/'evaluation-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()))
            raise
        atomic_json(child/'evaluation-terminal.json',dict(exit_code=0,finished_unix=time.time()))
        runs[cell['prompt']][cell['shard']]=str(child/'run')
        atomic_json(root/'completed-runs.json',runs)
        atomic_json(root/'progress.json',dict(phase='scored',done=index+1,total=len(expected),heartbeat_unix=time.time()))
    # Official scorer dependencies require the existing pinned scoring environment.
    first=root/expected[0]['prompt']/expected[0]['shard']
    launch=json.loads((first/'intent.json').read_text());source=first/'source'
    with (root/'comparison.log').open('xb') as log:
        subprocess.run([launch['scorer_python'],str(source/'tools/jev/compare_omni_slurp_transfer.py'),
            '--protocol',str(protocol_path),'--runs',str(root/'completed-runs.json'),
            '--repository',launch['scorer_repository'],'--output',str(root/'comparison.json')],
            env={**os.environ,'PYTHONPATH':str(source/'src')},cwd=source,stdout=log,stderr=log,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prepare',action='store_true')
    for name in ('protocol','validation-root','native-probe'):p.add_argument('--'+name,type=Path)
    p.add_argument('--devices',choices=['4,5','6,7']);a=p.parse_args()
    if a.prepare:
        if not all((a.protocol,a.validation_root,a.native_probe,a.devices)):p.error('preparation arguments required')
        print(json.dumps(prepare(a.root,a.protocol,a.validation_root,a.native_probe,list(map(int,a.devices.split(',')))),indent=2))
    else:
        with (a.root/'queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (a.root/'evaluation-terminal.json').exists():raise FileExistsError('existing queue terminal')
            try:run(a.root)
            except BaseException as exc:
                atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time(),automatic_retry=False))
                raise
            atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=0,finished_unix=time.time()))

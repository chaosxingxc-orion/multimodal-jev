#!/usr/bin/env python3
"""One Jev capacity attempt after independently accepted native capacity."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

import watch_audio2tool_capacity as native
from multimodal_jev.audio2tool_inference import write_once
from multimodal_jev.audio2tool_jev_capacity import require_capacity
from multimodal_jev.baseline import atomic_json, file_hash

PREPARATION_SHA='acbdcce6d4862cc85c84e97d35458b4a1abd69db7ad31f26434f8c4b92fd7712'
NATIVE_PLAN_SHA='66e8a43a7d4fae5dcd1e00d827149cbc281138fbdcd59018000f5050ffd8d4be'
NATIVE_LAUNCH_SHA='0c0989e897abbd86e05a05c2e1479ea3f10c10c9363159cf08121ae3dc5cb473'
RELEASE='/mnt/cephfs/xingchao/datasets/audio2tool-jev/a3a34fdd0da98e6f6c53e9201751993e37202f931a1240a7475bf90593c50235'
REGISTRY='/mnt/cephfs/xingchao/datasets/audio2tool/tools_registry.csv'
OBSERVATION_ERRORS=(subprocess.TimeoutExpired,subprocess.CalledProcessError,json.JSONDecodeError)


def native_ready(root):
    root=Path(root)
    if file_hash(root/'queue-plan.json')!=NATIVE_PLAN_SHA or file_hash(root/'evaluation-launch.json')!=NATIVE_LAUNCH_SHA:
        raise ValueError('registered native capacity queue required')
    launch=native.read(root/'evaluation-launch.json');terminal=root/'evaluation-terminal.json'
    if not terminal.exists():
        try:stat=Path(f"/proc/{launch['pid']}/stat").read_text().rsplit(')',1)[1].split()
        except FileNotFoundError:stat=[]
        if len(stat)<20 or stat[19]!=launch['process_start'] or stat[0]=='Z':
            raise RuntimeError('native capacity supervisor missing/replaced/dead without terminal')
        return False
    if native.read(terminal)['exit_code']!=0:raise RuntimeError('native capacity dependency failed')
    _,prep=native.check_pins(root,NATIVE_PLAN_SHA)
    if not native.sampling_ready(prep['dependency_root']):raise ValueError('native sampling dependency not accepted')
    cid=native.read(root/'launch.json')['container_id'];item=native.inspect(cid)
    # A fresh actual inspection is checked on every completed-dependency call.
    # Never use the existence of an acceptance file as proof of execution.
    with tempfile.TemporaryDirectory(prefix='jev-capacity-inspection-') as folder:
        path=Path(folder)/'inspection.json';write_once(path,item)
        receipt=native.require_capacity(root,path,file_hash(path),cid,root.parent/'audio2tool-native-processor-v1/run')
    accepted=native.read(root/'acceptance.json')
    for k in ('container_id','result_sha256','preparation_sha256'):
        if receipt[k]!=accepted[k]:raise ValueError('native capacity acceptance changed')
    return True


def check_pins(root,plan_hash):
    if file_hash(root/'queue-plan.json')!=plan_hash or file_hash(root/'preparation.json')!=PREPARATION_SHA:
        raise ValueError('registered Jev capacity plan required')
    queue=native.read(root/'queue-plan.json');prep=native.read(root/'preparation.json')
    if (queue['schema']!='audio2tool-jev-capacity-queue-v1' or queue['automatic_retry'] is not False
            or queue['dependency_seconds']!=350000 or queue['observation_seconds']!=3900
            or queue['resource_lock']!=str(root.parent/'npu67-sampling-resource.lock')
            or queue['dependency_root']!=prep['dependency_root']):
        raise ValueError('bounded sequential Jev queue required')
    for name,pin in queue['source_files'].items():
        if file_hash(root/'queue-source'/name)!=pin:raise ValueError('coordinator source changed')
    source=Path(prep['source_root']);preflight=source.parent
    for name,pin in prep['source_files'].items():
        if file_hash(source/name)!=pin:raise ValueError('original Jev probe source changed')
    if (file_hash(preflight/'intent.json')!=prep['preflight_intent_sha256']
            or file_hash(preflight/'run/result.json')!=prep['preflight_result_sha256']):
        raise ValueError('accepted Jev preflight changed')
    native.successful(prep['preflight_container_id'],prep['image'])
    return queue,prep


def dispatch(root,prep):
    # Persist the complete intent before create, including the input source
    # manifest required by the independent Jev capacity acceptance module.
    write_once(root/'intent.json',prep);intent_hash=file_hash(root/'intent.json')
    try:created=native.remote(prep['argv'],timeout=90)
    except OBSERVATION_ERRORS as exc:
        write_once(root/'create-error.json',dict(error=repr(exc)));raise
    cid=created.stdout.strip()
    write_once(root/'create-result.json',dict(stdout=created.stdout,stderr=created.stderr))
    if not re.fullmatch('[0-9a-f]{64}',cid):raise RuntimeError('ambiguous create; no replay')
    write_once(root/'launch.json',dict(container_id=cid,image=prep['image'],source_commit=prep['source_commit'],
                                     intent_sha256=intent_hash,physical_devices=[6,7]))
    item=native.inspect(cid)
    if item['Id']!=cid or item['Image']!=prep['image'] or item['State']['Status']!='created':
        raise ValueError('unexpected created Jev capacity container')
    write_once(root/'start-intent.json',dict(container_id=cid,started_unix=time.time(),automatic_retry=False))
    try:started=native.remote(['docker','start',cid],timeout=90)
    except OBSERVATION_ERRORS as exc:
        write_once(root/'start-error.json',dict(container_id=cid,error=repr(exc)));raise
    write_once(root/'start-result.json',dict(stdout=started.stdout,stderr=started.stderr))
    return cid,intent_hash


def free_devices(root):
    if not native.free_devices():return False
    # Also check driver-visible processes outside Docker. Unknown/nonempty
    # output is conservatively busy and retained for subsequent inspection.
    observed={str(i):native.remote(['npu-smi','info','-t','proc-mem','-i',str(i),'-c','0']).stdout
              for i in (6,7)}
    atomic_json(root/'device-process-observation.json',dict(devices=observed,observed_unix=time.time()))
    return all(not text.strip() for text in observed.values())


def execute(root,plan_hash):
    if any((root/n).exists() for n in ('intent.json','launch.json','run','evaluation-terminal.json')):
        raise FileExistsError('prior Jev capacity attempt; no replay')
    queue,prep=check_pins(root,plan_hash);deadline=time.monotonic()+queue['dependency_seconds']
    while True:
        try:
            ready=native_ready(queue['dependency_root'])
            atomic_json(root/'queue-progress.json',dict(phase='waiting_accepted_native_capacity',ready=ready,heartbeat_unix=time.time()))
            if ready:break
        except OBSERVATION_ERRORS as exc:
            atomic_json(root/'queue-progress.json',dict(phase='dependency_observation_retry',error_type=type(exc).__name__,heartbeat_unix=time.time()))
        if time.monotonic()>deadline:raise TimeoutError('dependency observation expired; no dispatch')
        time.sleep(20)
    with Path(queue['resource_lock']).open('a') as lock:
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:locked=False
            else:locked=True
            try:free=locked and free_devices(root)
            except OBSERVATION_ERRORS:free=False
            if free:break
            if locked:fcntl.flock(lock,fcntl.LOCK_UN)
            atomic_json(root/'queue-progress.json',dict(phase='waiting_npu67',heartbeat_unix=time.time()))
            if time.monotonic()>deadline:raise TimeoutError('resource observation expired; no dispatch')
            time.sleep(20)
        queue,prep=check_pins(root,plan_hash)
        if not native_ready(queue['dependency_root']):raise ValueError('native acceptance changed')
        cid,intent_hash=dispatch(root,prep);deadline=time.monotonic()+queue['observation_seconds']
        while True:
            try:
                item=native.inspect(cid)
                if item['Id']!=cid or item['Image']!=prep['image']:raise ValueError('capacity identity changed')
                atomic_json(root/'queue-progress.json',dict(phase='capacity_running',container_id=cid,state=item['State'],heartbeat_unix=time.time()))
                if item['State']['Status'] in ('exited','dead'):break
            except OBSERVATION_ERRORS as exc:
                atomic_json(root/'queue-progress.json',dict(phase='observation_retry',container_id=cid,error_type=type(exc).__name__,heartbeat_unix=time.time()))
            if time.monotonic()>deadline:raise TimeoutError('observation expired; inspect same ID, never replay')
            time.sleep(20)
        inspection=root/'capacity-inspection.json';write_once(inspection,item)
        receipt=require_capacity(root,intent_hash,inspection,file_hash(inspection),cid,
            root.parent/'audio2tool-jev-context-v1/run',RELEASE,Path(REGISTRY).read_bytes())
        native.successful(cid,prep['image']);write_once(root/'acceptance.json',receipt)
        atomic_json(root/'queue-progress.json',dict(phase='capacity_accepted',container_id=cid,heartbeat_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan-hash',required=True)
    args=p.parse_args()
    with (args.root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.root/'evaluation-terminal.json').exists():raise FileExistsError('preserve terminal')
        try:execute(args.root,args.plan_hash)
        except BaseException as exc:
            write_once(args.root/'evaluation-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time(),automatic_retry=False));raise
        write_once(args.root/'evaluation-terminal.json',dict(exit_code=0,finished_unix=time.time(),automatic_retry=False))

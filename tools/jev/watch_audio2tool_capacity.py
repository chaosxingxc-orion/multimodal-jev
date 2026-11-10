#!/usr/bin/env python3
"""One capacity attempt after accepted native SFC and K2 sampling queues."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

from multimodal_jev.audio2tool_inference import CAPACITY_PREPARATION_SHA,require_capacity,write_once
from multimodal_jev.baseline import atomic_json,file_hash

SAMPLING_PLAN_SHA='0fa10c9a99d8ee97d60e5a48675b6eef5b4e798c79a00df2db1dd140e4d34b79'
SAMPLING_LAUNCH_SHA='9c2f8e9e690843143b50f2192dd6ad69bf054a42960fa64c18bb176920ed7670'


def read(path):return json.loads(Path(path).read_text())


def remote(argv,timeout=30):
    return subprocess.run(['ssh','-o','ConnectTimeout=8','npu',shlex.join(argv)],
                          capture_output=True,text=True,timeout=timeout,check=True)


def inspect(cid):
    if not re.fullmatch('[0-9a-f]{64}',cid):raise ValueError('exact container ID required')
    return json.loads(remote(['docker','inspect',cid]).stdout)[0]


def successful(cid,image):
    item=inspect(cid);state=item['State']
    if (item['Id']!=cid or item['Image']!=image or state['Status']!='exited'
            or state['ExitCode']!=0 or state['OOMKilled']):
        raise ValueError('successful exact dependency container required')
    return item


def sampling_ready(root):
    root=Path(root)
    if file_hash(root/'queue-plan.json')!=SAMPLING_PLAN_SHA or file_hash(root/'evaluation-launch.json')!=SAMPLING_LAUNCH_SHA:
        raise ValueError('registered sampling queue changed')
    launch=read(root/'evaluation-launch.json');plan=read(root/'queue-plan.json')
    terminal=root/'evaluation-terminal.json'
    if not terminal.exists():
        try:start=Path(f"/proc/{launch['pid']}/stat").read_text().rsplit(')',1)[1].split()[19]
        except FileNotFoundError:start=None
        if start!=launch['process_start']:
            raise RuntimeError('sampling supervisor missing/replaced without terminal; no capacity dispatch')
        return False
    if read(terminal)['exit_code']!=0:raise RuntimeError('sampling dependency failed; no capacity dispatch')
    for name,pin in plan['files'].items():
        if file_hash(root/name)!=pin:raise ValueError('sampling artifact identity changed')
    preparation=read(root/'preparation.json')
    for name,pin in preparation['source_files'].items():
        if file_hash(root/'source'/name)!=pin:raise ValueError('sampling source changed')
    sample=read(root/'sampling-launch.json');validator=read(root/'acceptance-launch.json')
    item=successful(sample['container_id'],plan['image'])
    validator_item=successful(validator['container_id'],plan['image'])
    expected=preparation['argv'];expected=expected[expected.index(plan['image'])+1:]
    if item['Config']['Cmd']!=expected:raise ValueError('sampling command differs')
    saved_inspection=root/'sampling-inspection.json'
    expected_validator=plan['acceptance_argv'];expected_validator=expected_validator[expected_validator.index(plan['image'])+1:]
    expected_validator+=['--inspection',str(saved_inspection),'--inspection-hash',file_hash(saved_inspection)]
    if validator_item['Config']['Cmd']!=expected_validator:raise ValueError('actual acceptance command differs')
    acceptance=read(root/'acceptance.json');result=read(root/'run/result.json')
    if (acceptance['container_id']!=sample['container_id'] or acceptance['image']!=plan['image']
            or acceptance['source_commit']!=plan['source_commit'] or acceptance['responses']!=156
            or acceptance['weight_updates']!=0 or acceptance['result_sha256']!=file_hash(root/'run/result.json')
            or acceptance['records_sha256']!=file_hash(root/'run/records.jsonl')
            or result['status']!='passed' or result['model_loaded'] is not True
            or read(root/'run/terminal.json')['exit_code']!=0):
        raise ValueError('complete accepted sampling dependency required')
    native=Path(plan['dependency']['root'])
    if read(native/'evaluation-terminal.json')['exit_code']!=0:raise ValueError('native SFC queue unsuccessful')
    for child in plan['dependency']['container_children']:
        receipt=read(native/child/'launch.json');successful(receipt['container_id'],plan['image'])
    return True


def check_pins(root,plan_hash):
    if file_hash(root/'queue-plan.json')!=plan_hash or file_hash(root/'preparation.json')!=CAPACITY_PREPARATION_SHA:
        raise ValueError('registered capacity plan changed')
    queue=read(root/'queue-plan.json');prep=read(root/'preparation.json')
    if (queue['schema']!='audio2tool-capacity-queue-v1' or queue['automatic_retry'] is not False
            or queue['dependency_seconds']!=350000 or queue['observation_seconds']!=3900
            or queue['resource_lock']!=str(root.parent/'npu67-sampling-resource.lock')
            or queue['sampling_root']!=prep['dependency_root']):
        raise ValueError('registered bounded sequential queue required')
    for name,pin in queue['source_files'].items():
        if file_hash(root/'queue-source'/name)!=pin:raise ValueError('coordinator source changed')
    for name,pin in prep['source_files'].items():
        if file_hash(Path(prep['source_root'])/name)!=pin:raise ValueError('capacity source changed')
    preflight=Path(prep['source_root']).parent
    if file_hash(preflight/'run/result.json')!=prep['preflight_result_sha256']:
        raise ValueError('CPU preflight result changed')
    successful(prep['preflight_container_id'],prep['image'])
    successful(prep['processor_container_id'],prep['image'])
    return queue,prep


def free_devices():
    for cid in remote(['docker','ps','--no-trunc','-q']).stdout.split():
        item=inspect(cid);host=item['HostConfig']
        if host['Privileged'] or any(d['PathOnHost'] in ('/dev/davinci6','/dev/davinci7') for d in host.get('Devices') or []):
            return False
    return True


def dispatch(root,prep):
    # A create timeout or lost response is ambiguous. Keep its intent and never
    # submit create/start again automatically, even if no launch receipt exists.
    write_once(root/'dispatch-intent.json',dict(argv=prep['argv'],started_unix=time.time(),automatic_retry=False))
    try:created=remote(prep['argv'],timeout=90)
    except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as exc:
        write_once(root/'create-error.json',dict(error=repr(exc),stdout=str(getattr(exc,'stdout',None)),stderr=str(getattr(exc,'stderr',None))))
        raise
    cid=created.stdout.strip()
    (root/'create.log').write_text(created.stdout+created.stderr)
    if not re.fullmatch('[0-9a-f]{64}',cid):raise RuntimeError('ambiguous create; inspect retained intent')
    write_once(root/'launch.json',dict(container_id=cid,image=prep['image'],source_commit=prep['source_commit'],
        preparation_sha256=CAPACITY_PREPARATION_SHA,physical_devices=[6,7]))
    item=inspect(cid)
    if item['Image']!=prep['image'] or item['State']['Status']!='created':raise ValueError('unexpected created container')
    write_once(root/'start-intent.json',dict(container_id=cid,started_unix=time.time(),automatic_retry=False))
    try:started=remote(['docker','start',cid],timeout=90)
    except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as exc:
        write_once(root/'start-error.json',dict(container_id=cid,error=repr(exc),stdout=str(getattr(exc,'stdout',None)),stderr=str(getattr(exc,'stderr',None))))
        raise
    (root/'start.log').write_text(started.stdout+started.stderr)
    return cid


def execute(root,plan_hash):
    if any((root/n).exists() for n in ('dispatch-intent.json','launch.json','run','evaluation-terminal.json')):
        raise FileExistsError('prior capacity attempt; no replay')
    queue,prep=check_pins(root,plan_hash);deadline=time.monotonic()+queue['dependency_seconds']
    while True:
        try:
            ready=sampling_ready(queue['sampling_root'])
            atomic_json(root/'queue-progress.json',dict(phase='waiting_accepted_sampling',ready=ready,heartbeat_unix=time.time()))
            if ready:break
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,json.JSONDecodeError) as exc:
            atomic_json(root/'queue-progress.json',dict(phase='observation_retry',error_type=type(exc).__name__,heartbeat_unix=time.time()))
        if time.monotonic()>deadline:raise TimeoutError('dependency observation expired; no dispatch')
        time.sleep(20)
    with Path(queue['resource_lock']).open('a') as lock:
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:locked=False
            else:locked=True
            try:
                free=locked and free_devices()
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError,json.JSONDecodeError):free=False
            if free:break
            if locked:fcntl.flock(lock,fcntl.LOCK_UN)
            atomic_json(root/'queue-progress.json',dict(phase='waiting_npu67',heartbeat_unix=time.time()))
            if time.monotonic()>deadline:raise TimeoutError('resource observation expired; no dispatch')
            time.sleep(20)
        queue,prep=check_pins(root,plan_hash)
        if not sampling_ready(queue['sampling_root']):raise ValueError('dependency acceptance changed')
        cid=dispatch(root,prep);deadline=time.monotonic()+queue['observation_seconds']
        while True:
            try:
                item=inspect(cid)
                if item['Id']!=cid or item['Image']!=prep['image']:raise ValueError('capacity identity changed')
                atomic_json(root/'queue-progress.json',dict(phase='capacity_running',container_id=cid,state=item['State'],heartbeat_unix=time.time()))
                if item['State']['Status'] in ('exited','dead'):break
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError,json.JSONDecodeError) as exc:
                atomic_json(root/'queue-progress.json',dict(phase='observation_retry',container_id=cid,error_type=type(exc).__name__,heartbeat_unix=time.time()))
            if time.monotonic()>deadline:raise TimeoutError('capacity observation expired; inspect same ID, never replay')
            time.sleep(20)
        write_once(root/'capacity-inspection.json',item)
        logs=remote(['docker','logs',cid]);(root/'container.log').write_text(logs.stdout+logs.stderr)
        receipt=require_capacity(root,root/'capacity-inspection.json',file_hash(root/'capacity-inspection.json'),cid,
            root.parent/'audio2tool-native-processor-v1/run')
        write_once(root/'acceptance.json',receipt)
        successful(cid,prep['image'])
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

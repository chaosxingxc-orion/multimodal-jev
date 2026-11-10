#!/usr/bin/env python3
"""One original layer probe after the exact FSC NPU4/5 resource queue."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.audio2tool_inference import write_once

IMAGE = 'sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'
FSC_QUEUE = '2dcf2898dbdd805870e404bfb4589e14c286ac8654e090cb472e30f50c90aadb'
FSC_LAUNCH = '358f30e05241fb29f503c3508f71591deaee7a72418e9d12ff6b462075374ab3'
FSC_IDS = ['ae16d2e146ecab6207ac5779b6b561336a6d8ef941a83e4e279c9fde4526ebb0',
           '6f6fe878a1cbf6906cc679a5f2abcdb6403075cffc23b692f90bb7dbb03d957a']
OBSERVATION_ERRORS = (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError)


def read(path):
    return json.loads(Path(path).read_text())


def remote(argv, timeout=30):
    return subprocess.run(['ssh', '-o', 'ConnectTimeout=8', 'npu', shlex.join(argv)],
                          text=True, capture_output=True, check=True, timeout=timeout)


def inspect(cid):
    if not re.fullmatch('[0-9a-f]{64}', cid):
        raise ValueError('exact container ID required')
    return json.loads(remote(['docker', 'inspect', cid]).stdout)[0]


def live_process(launch):
    proc = Path('/proc')/str(launch['pid'])
    try:
        stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        argv = (proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
    except FileNotFoundError:
        return False
    return stat[0] != 'Z' and stat[19] == launch['process_start'] and argv == launch['argv']


def successful(item, cid):
    s = item['State']
    if (item['Id'] != cid or item['Image'] != IMAGE or s['Status'] != 'exited'
            or s['ExitCode'] != 0 or s['OOMKilled']):
        raise ValueError('successful exact original container required')


def dependency_ready(root):
    if file_hash(root/'queue-intent.json') != FSC_QUEUE or file_hash(root/'evaluation-launch.json') != FSC_LAUNCH:
        raise ValueError('original FSC queue identity changed')
    if not (root/'evaluation-terminal.json').exists():
        if not live_process(read(root/'evaluation-launch.json')):
            raise RuntimeError('original FSC supervisor missing without terminal; inspect, never replay')
        return False
    if read(root/'evaluation-terminal.json')['exit_code'] != 0:
        raise RuntimeError('FSC queue failed; inspect before dispatch')
    for child, cid in zip(('parent', 'p1'), FSC_IDS, strict=True):
        sub = root/child
        if read(sub/'launch.json')['container_id'] != cid:
            raise ValueError('FSC container replaced')
        item = inspect(cid)
        successful(item, cid)
        intent = read(sub/'intent.json');argv = intent['argv']
        if item['Config']['Cmd'] != argv[argv.index(IMAGE)+1:]:
            raise ValueError('FSC actual command changed')
        if read(sub/'run/terminal.json')['exit_code'] != 0 or read(sub/'evaluation-terminal.json')['exit_code'] != 0:
            raise ValueError('FSC execution not complete')
    # Only resource sequencing: do not consume labels or select by TEST score.
    return True


def check_container(item, prep, cid):
    if item['Id'] != cid or item['Image'] != IMAGE:
        raise ValueError('probe container identity changed')
    host = item['HostConfig'];argv = prep['argv']
    if (host['NanoCpus'] != 2*10**9 or host['Memory'] != 8*1024**3
            or host['MemorySwap'] != 8*1024**3 or host['NetworkMode'] != 'none' or host['Privileged']
            or {d['PathOnHost'] for d in host['Devices']} != {
                '/dev/davinci4', '/dev/davinci_manager', '/dev/devmm_svm', '/dev/hisi_hdc'}
            or any(d['PathOnHost'] != d['PathInContainer'] or d['CgroupPermissions'] != 'rwm' for d in host['Devices'])
            or item['Config']['Cmd'] != argv[argv.index(IMAGE)+1:]
            or set(host['Binds']) != {argv[i+1] for i,v in enumerate(argv[:-1]) if v == '-v'}
            or not {argv[i+1] for i,v in enumerate(argv[:-1]) if v == '-e'} <= set(item['Config']['Env'])):
        raise ValueError('actual probe devices, resources or command differ')


def check_pins(root, pin):
    if file_hash(root/'queue-plan.json') != pin:
        raise ValueError('registered queue plan changed')
    queue = read(root/'queue-plan.json')
    if file_hash(root/'preparation.json') != queue['preparation_sha256']:
        raise ValueError('probe preparation changed')
    prep = read(root/'preparation.json')
    if (prep['image'] != IMAGE or prep['physical_devices'] != [4] or prep['automatic_retry'] is not False
            or queue['dependency_seconds'] != 350000 or queue['observation_seconds'] != 900
            or queue['resource_lock'] != str(root.parent/'npu45-diagnostic-resource.lock')):
        raise ValueError('registered bounded one-card protocol required')
    for directory, files in [(root/'queue-source',queue['source_files']), (Path(prep['source_root']),prep['source_files'])]:
        for name, sha in files.items():
            if file_hash(directory/name) != sha:
                raise ValueError('immutable source changed')
    pre = Path(prep['preflight_root']);audit = read(pre/'diagnostic-audit.json')
    if (file_hash(pre/'diagnostic-audit.json') != prep['preflight_audit_sha256']
            or file_hash(pre/'run/result.json') != audit['result_sha256']
            or read(pre/'run/result.json')['npu_forward_calls'] != 0
            or prep['source_files'] != read(pre/'intent.json')['source_files']):
        raise ValueError('complete original load preflight required')
    successful(inspect(audit['container_id']), audit['container_id'])
    return queue, prep


def free_devices(root):
    for cid in remote(['docker','ps','--no-trunc','-q']).stdout.split():
        h = inspect(cid)['HostConfig']
        if h['Privileged'] or any(d['PathOnHost'] in ('/dev/davinci4','/dev/davinci5') for d in h.get('Devices') or []):
            return False
    observed = {str(i):remote(['npu-smi','info','-t','proc-mem','-i',str(i),'-c','0']).stdout for i in (4,5)}
    atomic_json(root/'device-observation.json',dict(observed=observed,time=time.time()))
    return all(not v.strip() for v in observed.values())


def dispatch(root, prep):
    write_once(root/'intent.json',prep)
    created = remote(prep['argv'],timeout=90)
    write_once(root/'create-result.json',dict(stdout=created.stdout,stderr=created.stderr))
    cid = created.stdout.strip()
    if not re.fullmatch('[0-9a-f]{64}',cid):
        raise RuntimeError('ambiguous create; no replay')
    write_once(root/'launch.json',dict(container_id=cid,image=IMAGE,source_commit=prep['source_commit'],
        intent_sha256=file_hash(root/'intent.json'),physical_devices=[4]))
    item = inspect(cid);check_container(item,prep,cid)
    if item['State']['Status'] != 'created':
        raise ValueError('expected unstarted container')
    write_once(root/'start-intent.json',dict(container_id=cid,time=time.time(),automatic_retry=False))
    started = remote(['docker','start',cid],timeout=90)
    write_once(root/'start-result.json',dict(stdout=started.stdout,stderr=started.stderr))
    return cid


def execute(root,pin):
    if any((root/n).exists() for n in ('intent.json','launch.json','run','evaluation-terminal.json')):
        raise FileExistsError('prior probe attempt; never replay')
    queue,prep = check_pins(root,pin);deadline = time.monotonic()+queue['dependency_seconds']
    dep = Path(queue['dependency_root'])
    while True:
        try:
            ready = dependency_ready(dep)
            atomic_json(root/'queue-progress.json',dict(phase='waiting_fsc_test_queue',ready=ready,heartbeat_unix=time.time()))
            if ready:break
        except OBSERVATION_ERRORS as exc:
            atomic_json(root/'queue-progress.json',dict(phase='dependency_observation_retry',error=type(exc).__name__,heartbeat_unix=time.time()))
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
            atomic_json(root/'queue-progress.json',dict(phase='waiting_npu45_free',heartbeat_unix=time.time()))
            if time.monotonic()>deadline:raise TimeoutError('resource observation expired; no dispatch')
            time.sleep(20)
        queue,prep=check_pins(root,pin)
        if not dependency_ready(dep):raise ValueError('resource dependency changed')
        cid=dispatch(root,prep);deadline=time.monotonic()+queue['observation_seconds']
        while True:
            try:
                item=inspect(cid);check_container(item,prep,cid)
                atomic_json(root/'queue-progress.json',dict(phase='probe_running',container_id=cid,state=item['State'],heartbeat_unix=time.time()))
                if item['State']['Status'] in ('exited','dead'):break
            except OBSERVATION_ERRORS as exc:
                atomic_json(root/'queue-progress.json',dict(phase='probe_observation_retry',container_id=cid,error=type(exc).__name__,heartbeat_unix=time.time()))
            if time.monotonic()>deadline:raise TimeoutError('observation expired; inspect same container, never replay')
            time.sleep(20)
        write_once(root/'final-inspection.json',item)
        logs=remote(['docker','logs',cid]);(root/'container.log').write_text(logs.stdout+logs.stderr)
        successful(item,cid)
        result=read(root/'run/result.json')
        if (read(root/'run/terminal.json')['exit_code'] != 0 or result['status'] != 'diagnostic_complete'
                or result['npu_forward_calls'] != 9 or result['preflight_only'] is not False
                or result['weight_updates'] != 0 or result['identity_sha256'] != file_hash(root/'run/identity.json')):
            raise ValueError('complete nine-call diagnostic required')
        write_once(root/'execution-complete.json',dict(container_id=cid,result_sha256=file_hash(root/'run/result.json'),
            scope='Execution complete only; independent tensors, numerical comparisons and capacity audit pending.'))
        atomic_json(root/'queue-progress.json',dict(phase='execution_complete_audit_pending',heartbeat_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan-hash',required=True)
    args=p.parse_args()
    with (args.root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.root/'evaluation-terminal.json').exists():raise FileExistsError('preserve terminal')
        try:execute(args.root,args.plan_hash)
        except BaseException as exc:
            write_once(args.root/'evaluation-terminal.json',dict(exit_code=1,error=repr(exc),automatic_retry=False,time=time.time()));raise
        write_once(args.root/'evaluation-terminal.json',dict(exit_code=0,automatic_retry=False,time=time.time()))

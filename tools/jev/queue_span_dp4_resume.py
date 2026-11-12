#!/usr/bin/env python3
"""Drain exact predecessors, then dispatch the prepared four-rank resume once.

Polling is ten minutes. A successful queue terminal means dispatched, not trained
or parity accepted. Unknown state fails closed; it never recreates a container.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.resource_observers import is_audited_completion_observer

SSH = ['ssh', '-o', 'ControlMaster=no', '-o', 'ControlPath=none', 'npu']
CARDS = (4, 5, 6, 7)


def remote(argv):
    return subprocess.run(SSH + [shlex.join(argv)], capture_output=True, text=True, timeout=60)


def inspect(ids):
    if not ids:
        return {}
    result = remote(['docker', 'inspect', *ids])
    result.check_returncode()
    rows = {r['Id']: r for r in json.loads(result.stdout)}
    if set(rows) != set(ids):
        raise ValueError('exact predecessor identities required')
    return rows


def check_pins(pins):
    for name, digest in pins.items():
        if file_hash(name) != digest:
            raise ValueError('queued asset changed: ' + name)


def dependencies(config):
    """Authenticate future evaluations through their original queue receipts."""
    ids = list(config['containers'])
    pending = []
    for entry in config['evaluation_queues']:
        root = Path(entry['root'])
        check_pins(entry['pins'])
        launch = json.loads((root / 'launch.json').read_text())
        terminal = root / 'run/terminal.json'
        dispatched = root / 'run/dispatched.json'
        if terminal.exists():
            if json.loads(terminal.read_text())['exit_code'] != 0:
                raise ValueError('predecessor evaluation queue failed: ' + str(root))
            if not dispatched.exists():
                raise ValueError('queue exited without evaluation identity')
            receipt = json.loads(dispatched.read_text())
            if receipt['evaluation'] != entry['evaluation']:
                raise ValueError('evaluation queue target changed')
            path = Path(entry['evaluation']) / 'launch.json'
            if file_hash(path) != receipt['launch_sha256']:
                raise ValueError('evaluation launch changed')
            actual = json.loads(path.read_text())
            if actual['container_id'] != receipt['container_id'] or actual['launch_exit_code'] != 0:
                raise ValueError('evaluation launch identity differs')
            ids.append(receipt['container_id'])
        else:
            # A stale queue file alone is not evidence of a live predecessor.
            path = Path('/proc') / str(launch['pid']) / 'cmdline'
            try:
                argv = path.read_bytes().rstrip(b'\0').decode().split('\0')
            except FileNotFoundError:
                # The process may have atomically published its terminal as we
                # read /proc. Re-evaluate on the next observation, never launch.
                if terminal.exists():
                    pending.append(str(root))
                    continue
                raise ValueError('predecessor queue missing without terminal: ' + str(root))
            if argv != launch['argv']:
                raise ValueError('predecessor queue PID was reused')
            pending.append(str(root))
    rows = inspect(list(dict.fromkeys(ids)))
    for cid, row in rows.items():
        state = row['State']
        if state['Status'] == 'exited':
            if state['ExitCode'] != 0:
                raise ValueError('predecessor failed: ' + cid)
        elif state['Status'] == 'running':
            pending.append(cid)
        else:
            raise ValueError('predecessor in unexpected state: ' + cid)
    return pending


FD_CHECK = '''import os,stat,json
cards={os.stat('/host-dev/davinci'+str(i)).st_rdev:i for i in (4,5,6,7)}
busy=[];denied=[]
for pid in os.listdir('/proc'):
 if not pid.isdigit():continue
 try:
  for fd in os.listdir('/proc/'+pid+'/fd'):
   try:
    s=os.stat('/proc/'+pid+'/fd/'+fd)
    if stat.S_ISCHR(s.st_mode) and s.st_rdev in cards:busy.append([int(pid),fd,cards[s.st_rdev]])
   except FileNotFoundError:pass
   except PermissionError:denied.append([pid,fd])
 except FileNotFoundError:pass
 except PermissionError:denied.append([pid,'fd'])
print(json.dumps(dict(busy=busy,denied=denied)))
raise SystemExit(1 if busy or denied else 0)
'''


def check_devices(image):
    result = remote(['docker', 'ps', '-q', '--no-trunc'])
    result.check_returncode()
    for row in inspect(result.stdout.split()).values():
        host = row['HostConfig']
        devices = {d['PathOnHost'] for d in host.get('Devices') or []}
        if (devices & {f'/dev/davinci{i}' for i in CARDS}
                or (host.get('Privileged') and not is_audited_completion_observer(row))):
            raise ValueError('physical cards reserved by container: ' + row['Id'])
    # Root read-only descriptor inspection also catches jobs outside Docker.
    # No --device mapping and no NPU open: this is an ownership check only.
    result = remote(['docker', 'run', '--rm', '--pull=never', '--network=none',
                     '--pid=host', '--user=0', '--cap-add=SYS_PTRACE',
                     '--security-opt', 'apparmor=unconfined', '--cpus=1', '--memory=1g',
                     '-v', '/dev:/host-dev:ro', image, 'python3', '-c', FD_CHECK])
    result.check_returncode()
    if json.loads(result.stdout) != {'busy': [], 'denied': []}:
        raise ValueError('host devices busy or descriptors inaccessible')


def prepared_plan(config):
    check_pins(config['pins'])
    plan = json.loads(Path(config['prepared_plan']).read_text())
    root = Path(plan['root'])
    if (root / 'launch.json').exists() or (root / 'run').exists():
        raise ValueError('resume already claimed; inspect it instead of relaunching')
    if plan['status'] != 'prepared_not_launched' or plan['physical_devices'] != list(CARDS):
        raise ValueError('prepared DP4 plan required')
    if plan['cpu_audit']['passed'] is not True:
        raise ValueError('native CPU resume audit required')
    check_pins({str(root/'config.json'): plan['config_sha256'],
                str(root/'source-manifest.json'): plan['source_manifest_sha256']})
    run_config = json.loads((root/'config.json').read_text())
    check_pins(run_config['pins'])
    manifest = json.loads((root/'source-manifest.json').read_text())
    check_pins({str(root/'source'/p): h for p, h in manifest.items()})
    argv = plan['argv']
    if argv[:3] != ['docker', 'run', '-d'] or argv.count(plan['image']) != 1:
        raise ValueError('unexpected prepared launch format')
    mapped = [argv[i+1] for i, x in enumerate(argv) if x == '--device']
    allowed = [f'/dev/{x}:/dev/{x}' for x in
               ('davinci4','davinci5','davinci6','davinci7','davinci_manager','devmm_svm','hisi_hdc')]
    if sorted(mapped) != sorted(allowed):
        raise ValueError('unexpected physical device mapping')
    return plan, ['docker', 'create', *argv[3:]]


def dispatch(config, output):
    with ExitStack() as stack:
        for card in CARDS:
            lock = stack.enter_context((Path(config['lock_directory']) / f'.physical-card{card}-launch.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if dependencies(config):
            raise ValueError('predecessors no longer ready')
        plan, argv = prepared_plan(config)
        root = Path(plan['root'])
        check_devices(plan['image'])
        # Publish the attempted exact argv BEFORE contacting Docker. Transport
        # timeout leaves a durable ambiguous attempt; never retry this queue.
        atomic_json(output/'create-attempt.json', {'argv': argv, 'time': time.time()})
        result = remote(argv)
        atomic_json(output/'create.json', dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        if result.returncode != 0 or not re.fullmatch('[a-f0-9]{64}', result.stdout.strip()):
            raise ValueError('creation failed or ambiguous; inspect exact prepared container name')
        cid = result.stdout.strip()
        receipt = {k: plan[k] for k in ('root','image','physical_devices','config_sha256','source_manifest_sha256')}
        receipt.update(argv=argv, container_id=cid, launch_exit_code=0, execution='created; start status recorded separately')
        atomic_json(root/'launch.json', receipt)
        atomic_json(output/'launch.json', receipt)
        check_devices(plan['image'])
        result = remote(['docker', 'start', cid])
        atomic_json(root/'start.json', dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        result.check_returncode()
        row = inspect([cid])[cid]
        if row['Image'] != plan['image'] or row['State']['Status'] not in ('running','exited'):
            raise ValueError('started container identity/state differs')
        if row['State']['Status'] == 'exited' and row['State']['ExitCode'] != 0:
            raise ValueError('resume failed immediately')
        atomic_json(output/'dispatched.json', dict(container_id=cid, root=str(root),
            scope='DP4 step-two recovery dispatched; independent HF/TP1 gate still required, not continuous verl training.'))


def run(config, output):
    prepared_plan(config)
    while True:
        check_pins(config['pins'])
        pending = dependencies(config)
        atomic_json(output/'progress.json', dict(stage='waiting_predecessors' if pending else 'dispatching',
                    pending=pending, updated_unix=time.time()))
        if not pending:
            dispatch(config, output)
            return
        time.sleep(600)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    try:
        run(json.loads(args.config.read_text()), args.output)
        atomic_json(args.output/'terminal.json', dict(exit_code=0, finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json', dict(exit_code=1, error=repr(exc), finished_unix=time.time()))
        raise

"""Launch the sealed finite parity diagnostic after the existing queue closes.

Default validates locally. Explicit launch retains the established device lease
through exact-container wait and collects outputs; observation errors never retry
training or recreate a container. This does not dispatch complex-task training.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from tools.reports.prepare_context_attribute_parity_npu import IMAGE, PREFIX, TAG, command, sha
from tools.reports.run_context_attribute_parity_pair import verify


def validate(prepared):
    prepared = Path(prepared)
    receipt = json.loads((prepared/'PREPARED.json').read_text())
    config = json.loads((prepared/'package/probe-config.json').read_text())
    root = PREFIX+TAG
    if (receipt['status'] != 'sealed_not_launched' or receipt['image'] != IMAGE
            or receipt['remote_root'] != root or receipt['training_dispatch_allowed'] is not False
            or receipt['source_manifest_sha256'] != verify(prepared/'package')
            or receipt['argv'] != command(root, config['model'])
            or config['physical_devices'] != [7] or config['source_root'] != root+'/package'):
        raise ValueError('Sealed finite diagnostic package required')
    return receipt, config


def launch(prepared):
    prepared = Path(prepared).resolve()
    receipt, config = validate(prepared)
    base = Path('.local/decision-objective-research')
    owner = (base/(TAG+'.lock')).open('a')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt, handle = (base/(TAG+s) for s in ('-attempt.json', '-launch.json'))
    if attempt.exists() or handle.exists():
        raise ValueError('Existing attempt or handle: reconcile, never recreate')
    predecessor = json.loads((base/'minds-context-arithmetic-v1-queue-status.json').read_text())
    if predecessor['phase'] != 'frozen_MINDS_collected_and_scored':
        raise ValueError('Existing simple baseline and evaluation queue has priority')
    lease = Path(receipt['lease']).open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    validate(prepared)
    def ssh(args):
        return subprocess.check_output(['ssh', '-o', 'ConnectTimeout=10', 'npu', shlex.join(args)],
                                       text=True, timeout=55)
    if json.loads(ssh(['docker', 'image', 'inspect', IMAGE]))[0]['Id'] != IMAGE:
        raise ValueError('Fixed image required')
    if 'No running processes found in NPU 7' not in ssh(['npu-smi', 'info']):
        raise ValueError('Device7 busy: preserve existing workload')
    root = receipt['remote_root']
    storage = Path('src/multimodal_jev/local_runtime_storage.py').read_text()
    ssh(['python3', '-c', storage+'\ncheck_paths({"root":"/data"})\n'
         +'assert not Path('+repr(root)+').exists()'])
    archive = base/(TAG+'-package.tar.gz')
    with tarfile.open(archive, 'x:gz') as tar:
        for path in sorted((prepared/'package').rglob('*')):
            if path.is_file():tar.add(path, arcname=str(path.relative_to(prepared)), recursive=False)
        tar.add(prepared/'package/probe-config.json', arcname='config.json', recursive=False)
    def write(path, value):
        with path.open('x') as f:json.dump(value, f, indent=2);f.write('\n')
    write(attempt, dict(prepared=str(prepared), receipt_sha256=sha(prepared/'PREPARED.json'),
                        remote_root=root, argv=receipt['argv']))
    ssh(['mkdir', '-p', root, root+'/home', root+'/cache'])
    subprocess.run(['scp', str(archive), 'npu:'+root+'/package.tar.gz'],
                   check=True, capture_output=True, timeout=55)
    ssh(['tar', '-xzf', root+'/package.tar.gz', '-C', root])
    check = ('import sys;sys.path.insert(0,'+repr(root+'/package')+');'
             'from tools.reports.run_context_attribute_parity_pair import verify;'
             'assert verify('+repr(root+'/package')+')=='+repr(receipt['source_manifest_sha256']))
    ssh(['python3', '-c', check])
    cid = ssh(receipt['argv']).strip()
    record = dict(container_id=cid, image=IMAGE, remote_root=root, argv=receipt['argv'],
                  source_manifest_sha256=receipt['source_manifest_sha256'],
                  config_sha256=sha(prepared/'package/probe-config.json'))
    write(handle, record)
    subprocess.run(['ssh', 'npu', shlex.join(['docker', 'start', cid])], capture_output=True)
    while True:
        result = subprocess.run(['ssh', 'npu', shlex.join(['docker', 'wait', cid])],
                                capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip().isdigit():break
        print('Observation failed; preserving exact handle and lease: '+cid, flush=True)
        for _ in range(10):time.sleep(60)
    from tools.jev.collect_context_rl_selection import collect
    collection = base/(TAG+'-collected')
    transport = collect(record, collection)
    observed = json.loads(ssh(['docker', 'inspect', cid]))[0]
    write(base/(TAG+'-inspect.json'), observed)
    terminal = collection/'evaluation/terminal.json'
    if (transport['state']['ExitCode'] != 0 or transport['state']['OOMKilled']
            or not terminal.is_file() or json.loads(terminal.read_text())['exit_code'] != 0):
        raise ValueError('Diagnostic failed; collected evidence preserved')
    print(json.dumps(dict(status='collected_independent_runtime_and_parity_audit_required',
                          container_id=cid, collection=str(collection))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.launch:launch(args.prepared)
    else:
        receipt, _ = validate(args.prepared)
        print(json.dumps(dict(status='validated_only_no_remote_calls',
                              source_manifest_sha256=receipt['source_manifest_sha256'],
                              argv=receipt['argv']), indent=2))

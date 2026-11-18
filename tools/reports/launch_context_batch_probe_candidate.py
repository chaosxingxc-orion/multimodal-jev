"""Launch one sealed finite probe on 175 after its existing queue completes.

Default is local validation. This does not add an automatic queue, relocate a
run to 181, or treat an observation timeout as permission to recreate a job.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from tools.reports.prepare_context_batch_probe_candidate import IMAGE, ROOT, command, sha
from tools.reports.run_context_batch_probe_candidate import verify

TAG = 'context-batch-probe-npu-v1'
LEASE = Path('.local/simple-final-v1/npu175-evaluation-training.lock')


def validate(prepared):
    prepared = Path(prepared)
    receipt = json.loads((prepared/'PREPARED.json').read_text())
    config = json.loads((prepared/'package/config.json').read_text())
    if (receipt['status'] != 'sealed_not_launched' or receipt['image'] != IMAGE
            or receipt['remote_root'] != ROOT or receipt['training_dispatch_allowed'] is not False
            or receipt['files_sha256'] != verify(prepared/'package')
            or receipt['argv'] != command(config['policy']['model'])
            or config['source_root'] != ROOT+'/package'
            or config['inputs'] != ROOT+'/package/inputs.pt'
            or config['policy']['physical_devices'] != [7]
            or config['policy']['max_batch_queries'] != 4
            or config['policy']['max_batch_tokens'] != 16384
            or config['policy'].get('process_device_scope') is not None):
        raise ValueError('sealed isolated-device batch4 probe required')
    if sha(prepared/'package/inputs.pt') != config['input_sha256']:
        raise ValueError('probe inputs changed')
    # Pins must cover the full payload except its own config/manifest.
    files = json.loads((prepared/'package/FILES.json').read_text())
    expected = {ROOT+'/package/'+n: h for n, h in files.items() if n != 'config.json'}
    if config['pins'] != expected:
        raise ValueError('complete runtime dependency pins required')
    return receipt, config


def check_priority(base):
    status = json.loads((Path(base)/'minds-context-arithmetic-v1-queue-status.json').read_text())
    if status['phase'] != 'frozen_MINDS_collected_and_scored':
        raise ValueError('existing simple training/evaluation queue has priority')


def check_device_observation(text):
    if 'No running processes found in NPU 7' not in text:
        raise ValueError('physical device7 is occupied or observation is unrecognized')


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2); f.write('\n')


def archive_package(prepared, destination):
    """Collector expects root/config.json, runner uses package/config.json."""
    prepared = Path(prepared)
    with tarfile.open(destination, 'x:gz') as tar:
        for p in sorted((prepared/'package').rglob('*')):
            if p.is_file(): tar.add(p, arcname=str(p.relative_to(prepared)), recursive=False)
        tar.add(prepared/'package/config.json', arcname='config.json', recursive=False)


def wait_exact(ssh, cid):
    while True:
        result = ssh(['docker', 'wait', cid], bounded=False)
        if result.returncode == 0 and result.stdout.strip().isdigit():
            return int(result.stdout.strip())
        print('Observation failed; keeping exact container and lease: '+cid, flush=True)
        # Background launcher only; ordinary observations remain >=600 seconds.
        for _ in range(10): time.sleep(60)


def launch(prepared):
    prepared = Path(prepared).resolve()
    receipt, config = validate(prepared)
    base = Path('.local/decision-objective-research')
    with (base/(TAG+'.lock')).open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt, handle = [base/(TAG+s) for s in ('-attempt.json', '-launch.json')]
        if attempt.exists() or handle.exists():
            raise ValueError('existing attempt/handle: reconcile the original, never recreate')
        check_priority(base)
        with LEASE.open('a') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            validate(prepared)
            def remote(args, bounded=True):
                return subprocess.run(['ssh', '-o', 'ConnectTimeout=10', 'npu', shlex.join(args)],
                                      text=True, capture_output=True, timeout=55 if bounded else None)
            def checked(args):
                r = remote(args)
                if r.returncode: raise RuntimeError(r.stderr)
                return r.stdout
            if json.loads(checked(['docker','image','inspect',IMAGE]))[0]['Id'] != IMAGE:
                raise ValueError('fixed image unavailable')
            check_device_observation(checked(['npu-smi','info']))
            storage = Path('src/multimodal_jev/local_runtime_storage.py').read_text()
            checked(['python3','-c',storage+'\ncheck_paths({"root":"/data"})\n'
                +'import shutil\nassert shutil.disk_usage("/data").free>20*1024**3\n'
                +'assert not Path('+repr(ROOT)+').exists()'])
            archive = base/(TAG+'-package.tar.gz')
            archive_package(prepared, archive)
            write_new(attempt, dict(prepared=str(prepared),prepared_sha256=sha(prepared/'PREPARED.json'),
                argv=receipt['argv'],host='npu',remote_root=ROOT,archive_sha256=sha(archive)))
            checked(['mkdir','-p',ROOT,ROOT+'/home',ROOT+'/cache'])
            subprocess.run(['scp',str(archive),'npu:'+ROOT+'/package.tar.gz'],
                           check=True,capture_output=True,timeout=55)
            checked(['tar','-xzf',ROOT+'/package.tar.gz','-C',ROOT])
            script='import sys;sys.path.insert(0,'+repr(ROOT+'/package')+');'
            script+='from tools.reports.run_context_batch_probe_candidate import verify;'
            script+='assert verify('+repr(ROOT+'/package')+')=='+repr(receipt['files_sha256'])
            checked(['python3','-c',script])
            cid = checked(receipt['argv']).strip()
            record = dict(container_id=cid,image=IMAGE,remote_root=ROOT,host='npu',argv=receipt['argv'],
                          config_sha256=sha(prepared/'package/config.json'),files_sha256=receipt['files_sha256'])
            write_new(handle, record)
            # Persist the exact handle before start; ambiguous start response is not a restart signal.
            try:
                start = remote(['docker','start',cid])
                write_new(base/(TAG+'-start-response.json'),dict(returncode=start.returncode,
                    stdout=start.stdout,stderr=start.stderr))
            except subprocess.TimeoutExpired:
                write_new(base/(TAG+'-start-response.json'),dict(observation_timeout=True))
            exit_code = wait_exact(remote, cid)
            write_new(base/(TAG+'-wait.json'),dict(container_id=cid,exit_code=exit_code))
            from tools.jev.collect_context_rl_selection import collect
            collection = base/(TAG+'-collected')
            transport = collect(record, collection, host='npu')
            write_new(base/(TAG+'-inspect.json'),json.loads(checked(['docker','inspect',cid]))[0])
            terminal = collection/'evaluation/terminal.json'
            if (transport['state']['ExitCode'] != 0 or transport['state']['OOMKilled']
                    or not terminal.is_file() or json.loads(terminal.read_text())['exit_code'] != 0):
                raise ValueError('probe failed; original logs and terminal collection preserved')
            print(json.dumps(dict(status='collected_independent_audit_required',collection=str(collection))))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--launch',action='store_true')
    a=p.parse_args()
    if a.launch:launch(a.prepared)
    else:
        receipt,_=validate(a.prepared)
        print(json.dumps(dict(status='validated_only_no_remote_calls',files_sha256=receipt['files_sha256'],
                              prerequisites=receipt['prerequisites']),indent=2))

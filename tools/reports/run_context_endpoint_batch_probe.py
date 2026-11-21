"""Launch and collect a finite endpoint probe under an existing host lease."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    with path.open('x') as stream:
        json.dump(obj, stream, indent=2)
        stream.write('\n')


def remote(host, args, **kwargs):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                           host, shlex.join([str(arg) for arg in args])],
                          capture_output=True, check=True, **kwargs).stdout


def inspect(host, cid):
    info, = json.loads(remote(host, ['docker', 'inspect', cid], timeout=30))
    if info['Id'] != cid:
        raise ValueError('poll only the exact owned container')
    return {key: info[key] for key in ('Id', 'Image', 'Name', 'State')}


def command(config, original, arm_name, arm):
    root = config['remote_root']
    parent = original['remote_root']
    tag = f"{config['tag']}-{arm_name}"
    args = ['docker', 'create', '--pull=never', '--network=none', '--name', tag,
            '--cpus=8', '--memory=20g', '--shm-size=4g']
    path = f"/dev/davinci{arm['device']}"
    for device in (path, '/dev/davinci_manager', '/dev/devmm_svm', '/dev/hisi_hdc'):
        args.extend(['--device', f'{device}:{device}'])
    for mount in (f'{parent}:{parent}:ro', f'{root}:{root}',
                  f'{root}/{arm_name}/cache:/jev-tmp', f'{root}/{arm_name}/home:/root'):
        args.extend(['-v', mount])
    prior = original['argv']
    for option, value in zip(prior, prior[1:]):
        if option == '-v':
            src = value.partition(':')[0]
            if not Path(src).is_relative_to(parent):
                if not value.endswith(':ro'):
                    raise ValueError(f'external mount must be read only: {src}')
                args.extend(['-v', value])
        if option == '-e':
            if value.startswith('PYTHONPATH=') and 'source_overlay' in config:
                overlay = f'{root}/overlay'
                value = f'PYTHONPATH={root}:{overlay}:{overlay}/src:{overlay}/tools/jev'
            args.extend(['-e', value])
    if 'source_overlay' in config:
        args.extend(['-v', f'{root}/overlay:{root}/overlay:ro'])
    args.extend(['-e', 'ASCEND_RT_VISIBLE_DEVICES=0', '-e', 'VLLM_RPC_BASE_PATH=/jev-tmp',
                 '--entrypoint', '/opt/multimodal-jev/runtime-entrypoint.sh', original['image'],
                 'python', f'{root}/probe.py', '--spec', f'{root}/spec.json',
                 '--arm', arm_name, '--output', f'{root}/{arm_name}/result'])
    return args


def run(config, out):
    host = config['host']
    original = read(config['parent_launch'])
    if original['host'] != host:
        raise ValueError('probe must use the same target host and pinned image')
    source = Path(config['probe'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != config['probe_sha256']:
        raise ValueError('probe source changed')
    arms = config['spec']['arms']
    devices = {arm['device'] for arm in arms.values()}
    if len(devices) != len(arms) or not all(re.fullmatch('[a-z0-9-]+', name) for name in arms):
        raise ValueError('require disjoint device assignments and safe tags')
    out.mkdir(exist_ok=False)
    write(out / 'config.json', config)
    root = config['remote_root']
    results = {}
    with Path(config['lease']).open() as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        image, = json.loads(remote(host, ['docker', 'image', 'inspect', original['image']], timeout=30))
        if image['Id'] != original['image']:
            raise ValueError('exact pinned image required')
        usage = remote(host, ['npu-smi', 'info'], timeout=30).decode()
        if any(f'No running processes found in NPU {d}' not in usage for d in devices):
            raise ValueError('require the declared NPUs to be free')
        remote(host, ['test', '-r', f"{original['remote_root']}/PREPARED.json"], timeout=30)
        remote(host, ['mkdir', root], timeout=30)
        if 'source_overlay' in config:
            subprocess.run(['scp', '-q', '-r', config['source_overlay'], f'{host}:{root}/overlay'],
                           timeout=90, check=True)
            remote(host, ['chmod', '-R', 'a-w', f'{root}/overlay'], timeout=30)
        for name in arms:
            remote(host, ['mkdir', '-p', f'{root}/{name}/cache', f'{root}/{name}/home'], timeout=30)
        (out / 'spec.json').write_text(json.dumps(config['spec'], indent=2) + '\n')
        for path, name in ((source, 'probe.py'), (out / 'spec.json', 'spec.json')):
            subprocess.run(['scp', '-q', str(path), f'{host}:{root}/{name}'], timeout=45, check=True)
        sha = remote(host, ['sha256sum', f'{root}/probe.py', f'{root}/spec.json'], timeout=30).decode().splitlines()
        if [line.split()[0] for line in sha] != [
                config['probe_sha256'], hashlib.sha256((out / 'spec.json').read_bytes()).hexdigest()]:
            raise ValueError('remote diagnostic source or spec changed')

        def execute(name, arm):
            argv = command(config, original, name, arm)
            cid = remote(host, argv, timeout=60).decode().strip()
            if not re.fullmatch('[0-9a-f]{64}', cid):
                raise ValueError('exact container ID required')
            write(out / f'{name}-launch.json', dict(container_id=cid, argv=argv))
            try:
                remote(host, ['docker', 'start', cid], timeout=60)
                code = int(remote(host, ['docker', 'wait', cid], timeout=config['deadline_seconds']))
            except BaseException:
                if inspect(host, cid)['State']['Running']:
                    remote(host, ['docker', 'stop', '-t', '30', cid], timeout=45)
                raise
            finally:
                info = inspect(host, cid)
                write(out / f'{name}-inspect.json', info)
                logs = subprocess.run(['ssh', '-o', 'BatchMode=yes', host,
                    shlex.join(['docker', 'logs', cid])], stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, timeout=60, check=True).stdout
                (out / f'{name}.log').write_bytes(logs)
            if info['Image'] != original['image'] or info['State']['Running'] or info['State']['ExitCode'] != code:
                raise ValueError('terminal container identity differs')
            files = remote(host, ['find', f'{root}/{name}/result', '-maxdepth', '1', '-type', 'f'],
                           timeout=30).decode().splitlines() if code == 0 else []
            receipt = {}
            if files:
                target = out / name
                target.mkdir()
                lines = remote(host, ['sha256sum', *files], timeout=60).decode().splitlines()
                subprocess.run(['scp', '-q', *[f'{host}:{path}' for path in files], str(target)],
                               timeout=180, check=True)
                for line in lines:
                    digest, path = line.split(maxsplit=1)
                    file = target / Path(path).name
                    with file.open('rb') as stream:
                        if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                            raise ValueError(f'collected evidence changed: {file}')
                    receipt[file.name] = digest
            return dict(container_id=cid, exit_code=code, evidence=receipt)

        with concurrent.futures.ThreadPoolExecutor(max_workers=config['parallel_arms']) as executor:
            tasks = {executor.submit(execute, name, arm): name for name, arm in arms.items()}
            for task in concurrent.futures.as_completed(tasks):
                name = tasks[task]
                try:
                    results[name] = task.result()
                except Exception as exc:
                    results[name] = dict(error=repr(exc), exit_code=1)
    code = int(any(r['exit_code'] != 0 for r in results.values()))
    write(out / 'terminal.json', dict(exit_code=code, arms=results, training_dispatched=False))
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(read(args.config), args.output))

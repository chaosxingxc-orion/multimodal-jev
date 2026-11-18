"""Launch a sealed SLURP TEST pair using the existing lease and exact-handle lifecycle.

Default is read-only preparation. No target scores, tuning or automatic retry.
Lifecycle reuses launch_context_rl_evaluation_v2; only package/command differ.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from tools.reports.prepare_slurp_context_test_package import verify, sha, IMAGE, PREFIX, CACHE


def command(manifest, config):
    remote = manifest['remote_root']
    if config['inputs']['cache_root'] != CACHE or config['physical_devices'] != [7]:
        raise ValueError('fixed audited cache and isolated device7 required')
    argv = ['docker', 'create', '--pull=never', '--network=none', '--name', Path(remote).name,
            '--cpus=8', '--memory=64g', '--shm-size=2g', '--log-driver=json-file',
            '--log-opt', 'max-size=20m', '--log-opt', 'max-file=2']
    for device in ('davinci7','davinci_manager','devmm_svm','hisi_hdc'):
        argv += ['--device','/dev/'+device+':/dev/'+device]
    mounts = [remote+':'+remote, remote+'/home:/root', remote+'/cache:/jev-tmp',
              CACHE+':'+CACHE+':ro', '/mnt/cephfs/xingchao:/mnt/cephfs/xingchao:ro',
              '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro']
    for name in ('source','frozen-source','inputs','processor','config.json','PACKAGE.json'):
        mounts.append(remote+'/'+name+':'+remote+'/'+name+':ro')
    for mount in mounts:argv += ['-v',mount]
    env = dict(PYTHONPATH=remote+'/source:'+remote+'/source/src:'+remote+'/source/tools/jev',
               PYTHONDONTWRITEBYTECODE='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2',
               HCCL_DETERMINISTIC='true', TMPDIR='/jev-tmp', VLLM_RPC_BASE_PATH='/jev-tmp')
    for key in ('HF_HOME','XDG_CACHE_HOME','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR',
                'ASCEND_WORK_PATH','ASCEND_CACHE_PATH'):env[key]='/jev-tmp'
    for key,value in env.items():argv += ['-e',key+'='+value]
    return argv+['--entrypoint','/opt/multimodal-jev/runtime-entrypoint.sh',IMAGE,'python',
                 remote+'/source/tools/reports/run_slurp_context_test_pair.py','--spec',remote+'/config.json',
                 '--output',remote+'/evaluation']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--launch', action='store_true')
    a = p.parse_args()
    m = verify(a.package, a.manifest_sha256)
    c = json.loads((a.package / 'config.json').read_text())
    argv = command(m, c)
    if not a.launch:
        print(json.dumps(dict(status='validated_only_no_remote_calls', argv=argv), indent=2));return
    root = Path('.local/decision-objective-research')
    tag = Path(m['remote_root']).name
    launch_path = root / (tag + '-launch.json')
    attempt_path = root / (tag + '-attempt.json')
    owner = (root / (tag + '.lock')).open('a')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if launch_path.exists() or attempt_path.exists():
        raise ValueError('existing launch or uncertain attempt; reconcile exact name/handle, never recreate')
    def ssh(args):
        return subprocess.check_output(['ssh', 'npu', shlex.join(args)], text=True, timeout=60)
    lease = Path('.local/simple-final-v1/npu175-evaluation-training.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX)
    verify(a.package, a.manifest_sha256)
    if json.loads(ssh(['docker', 'image', 'inspect', IMAGE]))[0]['Id'] != IMAGE:
        raise ValueError('fixed image unavailable')
    if 'No running processes found in NPU 7' not in ssh(['npu-smi', 'info']):
        raise ValueError('device7 busy; retain existing workload')
    remote = m['remote_root']
    if ssh(['python3', '-c', 'from pathlib import Path;print(Path(' + repr(remote) + ').exists())']).strip() != 'False':
        raise ValueError('remote output already exists; reconcile without overwrite')
    archive = root / (tag + '-package.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(a.package.rglob('*')):
            if path.is_file():tar.add(path, arcname=str(path.relative_to(a.package)), recursive=False)
    with attempt_path.open('x') as stream:
        json.dump(dict(container_name=tag, remote_root=remote, manifest_sha256=a.manifest_sha256), stream)
    ssh(['mkdir', '-p', remote, remote + '/cache', remote + '/home'])
    subprocess.run(['scp', str(archive), 'npu:' + remote + '/package.tar.gz'], check=True, timeout=60, capture_output=True)
    ssh(['tar', '-xzf', remote + '/package.tar.gz', '-C', remote])
    verify_remote = '''import hashlib,json
from pathlib import Path
r=Path(ROOT)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(r/'PACKAGE.json')==DIGEST
m=json.loads((r/'PACKAGE.json').read_text())
assert all(sha(r/p)==h for p,h in m['files'].items())
'''.replace('ROOT', repr(remote)).replace('DIGEST', repr(a.manifest_sha256))
    ssh(['python3', '-c', verify_remote])
    cid = ssh(argv).strip()
    launch = dict(container_id=cid, image=IMAGE, remote_root=remote, argv=argv,
        config_sha256=sha(a.package / 'config.json'), manifest_sha256=a.manifest_sha256)
    with launch_path.open('x') as stream:json.dump(launch, stream, indent=2)
    # Persist before start. Start observation failure never authorizes recreation.
    subprocess.run(['ssh', 'npu', shlex.join(['docker', 'start', cid])], capture_output=True)
    while True:
        result = subprocess.run(['ssh', 'npu', shlex.join(['docker', 'wait', cid])], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip().isdigit():
            print(json.dumps(dict(container_id=cid, exit_code=int(result.stdout.strip()),
                status='terminal_collection_required')), flush=True);return
        # Keep the lease through transient observation failures; exact wait is
        # retried only on the same handle, at the ordinary observation interval.
        print('Observation failed; retaining exact handle and lease: ' + cid, flush=True)
        for _ in range(10):time.sleep(60)


if __name__ == '__main__':main()

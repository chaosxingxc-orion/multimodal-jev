"""Launch a published formal package under the existing training/eval lease.

Default is local validation only. Exact handles persist before start; failed
observations retain the same handle and lease, never create another run.
"""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import time

from tools.jev.prepare_context_formal_package import verify, sha, IMAGE, PREFIX


def command(m, c):
    remote = m['remote_root'];cache = c['encoded_audio_cache'];mirror = c['audio_mirror_root']
    if (not re.fullmatch(re.escape(PREFIX)+r'context-rl-(grpo|rloo)-[a-z0-9-]+',remote)
            or not re.fullmatch(r'[0-9a-f]{64}',m['mount_source_container'])
            or c['physical_devices'] != [4,5,6,7]
            or cache != PREFIX+'context-learning-cache-v1/run/cache'
            or mirror != PREFIX+'context-learning-audio-mirror-v1/run'
            or c['start_update'] != 0 or c['stop_update'] != 634
            or c['engineering_prefix'] is not False or c['training_dispatch_allowed'] is not True):
        raise ValueError('fixed full-run topology/cache/root required')
    argv=['docker','create','--pull=never','--network=none','--name',Path(remote).name,
        '--cpus=24','--memory=128g','--shm-size=8g','--volumes-from',m['mount_source_container']+':ro',
        '--log-driver=json-file','--log-opt','max-size=20m','--log-opt','max-file=2']
    for device in ('davinci4','davinci5','davinci6','davinci7','davinci_manager','devmm_svm','hisi_hdc'):
        argv+=['--device','/dev/'+device+':/dev/'+device]
    mounts=[remote+':'+remote,remote+'/exchange:/jev-exchange',remote+'/home:/root',
        remote+'/cache:/jev-tmp',str(Path(cache).parent)+':'+str(Path(cache).parent)+':ro',
        cache+':'+cache,mirror+':'+mirror+':ro',
        '/mnt/cephfs/xingchao:/mnt/cephfs/xingchao:ro',
        '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro']
    for name in ('source','initial','evidence','run.py','config.json','service-config.json','RELEASE.json','PREPARED.json'):
        mounts.append(remote+'/'+name+':'+remote+'/'+name+':ro')
    for mount in mounts:argv+=['-v',mount]
    env=dict(PYTHONPATH=remote+'/source/src:'+remote+'/source/tools/jev',PYTHONDONTWRITEBYTECODE='1',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='2',HCCL_DETERMINISTIC='true',TMPDIR='/jev-tmp')
    for key in ('HF_HOME','XDG_CACHE_HOME','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','ASCEND_WORK_PATH','ASCEND_CACHE_PATH'):
        env[key]=remote+'/cache'
    for key,value in env.items():argv+=['-e',key+'='+value]
    return argv+['--entrypoint','/opt/multimodal-jev/runtime-entrypoint.sh',IMAGE,'python',remote+'/run.py']


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--launch',action='store_true')
    a=p.parse_args();m=verify(a.package,a.manifest_sha256)
    c=json.loads((a.package/'config.json').read_text());argv=command(m,c)
    if not a.launch:
        print(json.dumps(dict(status='validated_only_no_remote_calls',argv=argv),indent=2));return
    root=Path('.local/decision-objective-research');tag=Path(m['remote_root']).name
    attempt=root/(tag+'-attempt.json');launch_path=root/(tag+'-launch.json')
    owner=(root/(tag+'.lock')).open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if attempt.exists() or launch_path.exists():raise ValueError('existing attempt/handle; reconcile without recreation')
    lease=Path('.local/simple-final-v1/npu175-evaluation-training.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX)
    verify(a.package,a.manifest_sha256)
    def ssh(args):return subprocess.check_output(['ssh','npu',shlex.join(args)],text=True,timeout=60)
    if json.loads(ssh(['docker','image','inspect',IMAGE]))[0]['Id'] != IMAGE:
        raise ValueError('fixed image required')
    donor=json.loads(ssh(['docker','inspect',m['mount_source_container']]))[0]
    if (donor['Id'] != m['mount_source_container'] or donor['Image'] != IMAGE
            or donor['State']['Status'] != 'exited' or donor['State']['ExitCode'] != 0 or donor['State']['OOMKilled']):
        raise ValueError('successful original shared-container mount source required')
    usage=ssh(['npu-smi','info'])
    if any(f'No running processes found in NPU {i}' not in usage for i in (4,5,6,7)):
        raise ValueError('leased devices busy; retain current workload')
    remote=m['remote_root']
    if ssh(['python3','-c','from pathlib import Path;print(Path('+repr(remote)+').exists())']).strip() != 'False':
        raise ValueError('remote output exists; do not overwrite')
    archive=root/(tag+'-package.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for path in sorted(a.package.rglob('*')):
            if path.is_file():tar.add(path,arcname=str(path.relative_to(a.package)),recursive=False)
    with attempt.open('x') as stream:json.dump(dict(argv=argv,remote_root=remote,manifest_sha256=a.manifest_sha256),stream)
    ssh(['mkdir','-p',remote,remote+'/cache',remote+'/home',remote+'/exchange'])
    subprocess.run(['scp',str(archive),'npu:'+remote+'/package.tar.gz'],check=True,capture_output=True,timeout=60)
    ssh(['tar','-xzf',remote+'/package.tar.gz','-C',remote])
    check='''import hashlib,json
from pathlib import Path
r=Path(ROOT)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(r/'PREPARED.json')==DIGEST
m=json.loads((r/'PREPARED.json').read_text())
assert all(sha(r/p)==h for p,h in m['files'].items())
'''.replace('ROOT',repr(remote)).replace('DIGEST',repr(a.manifest_sha256))
    ssh(['python3','-c',check])
    cid=ssh(argv).strip()
    launch=dict(container_id=cid,image=IMAGE,remote_root=remote,argv=argv,
        config_sha256=sha(a.package/'config.json'),service_config_sha256=sha(a.package/'service-config.json'),
        manifest_sha256=a.manifest_sha256,release_id=m['release_id'],algorithm=m['algorithm'])
    with launch_path.open('x') as stream:json.dump(launch,stream,indent=2)
    subprocess.run(['ssh','npu',shlex.join(['docker','start',cid])],capture_output=True)
    while True:
        result=subprocess.run(['ssh','npu',shlex.join(['docker','wait',cid])],capture_output=True,text=True)
        if result.returncode==0 and result.stdout.strip().isdigit():
            print(json.dumps(dict(container_id=cid,exit_code=int(result.stdout.strip()),
                                  status='terminal_collection_required')),flush=True);return
        print('Observation failed; keeping exact handle and lease: '+cid,flush=True)
        for _ in range(10):time.sleep(60)


if __name__=='__main__':main()

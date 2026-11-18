"""Finite original-RLOO recovery operations; never stop or replace a job.

Uses the existing exclusive lease, exact wait and terminal collector. Default
validation is local. Only two recovery updates are admitted, not full training.
"""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.jev.collect_context_rl_training import collect
from tools.reports.context_arithmetic_resume_runtime import expected_config
from tools.reports.context_dual_training_operations import IMAGE, path
from tools.reports.launch_context_batch_probe_candidate import wait_exact
from tools.reports.prepare_context_dual_throughput_pair import sha, write


def validate(c):
    root = Path(c['package']); manifest = json.loads((root/'PREPARED.json').read_text())
    if sha(root/'PREPARED.json') != c['manifest_sha256']:
        raise ValueError('immutable recovery manifest required')
    actual = {str(p.relative_to(root)):sha(p) for p in root.rglob('*')
              if p.is_file() and p != root/'PREPARED.json'}
    if any(p.is_symlink() for p in root.rglob('*')) or actual != manifest['files']:
        raise ValueError('complete unchanged recovery package required')
    if (manifest['status'] != 'arithmetic_recovery_prefix_prepared_only' or manifest['image'] != IMAGE
            or manifest['training_dispatch_allowed'] is not False or manifest['finite_dispatch_allowed'] is not False):
        raise ValueError('finite prepared original recovery package required')
    config = json.loads((root/'config.json').read_text()); service = json.loads((root/'service-config.json').read_text())
    binding = json.loads((root/'RESUME.json').read_text())
    original_root = Path(c['original_package'])
    if sha(original_root/'PREPARED.json') != binding['original_manifest_sha256']:
        raise ValueError('original package identity differs')
    original = json.loads((original_root/'config.json').read_text())
    if sha(original_root/'config.json') != binding['original_config_sha256']:
        raise ValueError('original config identity differs')
    remote = path(manifest['remote_root']); origin = path(config['resume_origin_config'])
    if (binding['runtime_root'] != remote or binding['training_dispatch_allowed'] is not False
            or binding['schema'] != 'context-arithmetic-recovery-prefix-v1'):
        raise ValueError('recovery binding differs from package runtime')
    expected = expected_config(original, origin_config=origin, remote_root=remote, step=binding['step'])
    if ({k:v for k,v in config.items() if k != 'pins'} != expected
            or config['algorithm'] != 'rloo_v1' or config['physical_devices'] != [4,5,6,7]
            or config['global_batch'] != 128 or config['microbatch'] != 8
            or config['resume_root'] != str(Path(origin).parent/'run')):
        raise ValueError('unchanged objective and exactly two recovery updates required')
    expected_service = json.loads((original_root/'service-config.json').read_text())
    expected_service['pins'] = config['pins']
    expected_service['policy'].update(snapshot=config['resume_root']+f'/policy-{binding["step"]:04d}',
        adapter=config['initial_adapter'], policy_id=binding['behavior_policy_id'])
    if service != expected_service:raise ValueError('original inference runtime and recovered policy required')
    for name,digest in config['pins'].items():
        path(name)
        if name.startswith(remote+'/') and actual.get(name[len(remote)+1:]) != digest:
            raise ValueError('internal recovery dependency changed')
    if not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]*',c['host']):raise ValueError('safe host alias required')
    if not re.fullmatch('[a-z0-9][a-z0-9-]*',c['tag']):raise ValueError('safe unique tag required')
    return manifest, config


def command(c, manifest, config):
    remote = path(manifest['remote_root']); original = str(Path(config['resume_origin_config']).parent)
    shared = path(c['shared_root']); driver = path(c['driver_root'])
    cache = path(config['encoded_audio_cache']); mirror = path(config['audio_mirror_root'])
    inputs = [original, shared, driver, str(Path(cache).parent), mirror]
    if any(Path(x).is_relative_to(remote) or Path(remote).is_relative_to(x) for x in inputs):
        raise ValueError('recovery outputs and original reusable inputs must be disjoint')
    mounts = [(remote,remote,False),(remote+'/exchange','/jev-exchange',False),
        (remote+'/home','/root',False),(remote+'/cache','/jev-tmp',False),
        (original,original,True),(shared,shared,True),(driver,'/usr/local/Ascend/driver',True),
        (str(Path(cache).parent),str(Path(cache).parent),True),(cache,cache,False),(mirror,mirror,True)]
    for name in ('source','run.py','config.json','service-config.json','RESUME.json','CPU-RECOVERY.json','PREPARED.json'):
        mounts.append((remote+'/'+name,remote+'/'+name,True))
    if any(not any(Path(n).is_relative_to(dst) for _,dst,_ in mounts) for n in config['pins']):
        raise ValueError('dependency outside explicit mounts')
    argv = ['docker','create','--pull=never','--network=none','--name',c['tag'],
        '--cpus=24','--memory=128g','--shm-size=8g','--log-driver=json-file',
        '--log-opt','max-size=20m','--log-opt','max-file=2']
    for name in ('davinci4','davinci5','davinci6','davinci7','davinci_manager','devmm_svm','hisi_hdc'):
        argv += ['--device','/dev/'+name+':/dev/'+name]
    for src,dst,readonly in mounts:argv += ['-v',src+':'+dst+(':ro' if readonly else '')]
    env = dict(PYTHONPATH=remote+'/source/src:'+remote+'/source/tools/jev',PYTHONDONTWRITEBYTECODE='1',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='2',HCCL_DETERMINISTIC='true',TMPDIR='/jev-tmp')
    for name in ('HF_HOME','XDG_CACHE_HOME','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','ASCEND_WORK_PATH','ASCEND_CACHE_PATH'):
        env[name]=remote+'/cache'
    for name,value in env.items():argv += ['-e',name+'='+value]
    return argv+['--entrypoint','/opt/multimodal-jev/runtime-entrypoint.sh',IMAGE,'python',remote+'/run.py']


def launch(c, provenance):
    manifest,config=validate(c);argv=command(c,manifest,config)
    if c.get('finite_recovery_dispatch_allowed') is not True:
        raise ValueError('finite recovery dispatch not enabled')
    base=Path(c['operations_root']);check_paths({'operations':base});base.mkdir(parents=True,exist_ok=True)
    with (base/(c['tag']+'.lock')).open('a') as owner,Path(c['lease']).open('r') as lease:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempt=base/(c['tag']+'-attempt.json');handle=base/(c['tag']+'-launch.json')
        if attempt.exists() or handle.exists():raise ValueError('existing attempt: reconcile exact handle')
        def remote(args,bounded=True):
            return subprocess.run(['ssh','-o','ConnectTimeout=10',c['host'],shlex.join(args)],
                capture_output=True,text=True,timeout=55 if bounded else None)
        def checked(args):
            result=remote(args)
            if result.returncode:raise RuntimeError(result.stderr)
            return result.stdout
        if json.loads(checked(['docker','image','inspect',IMAGE]))[0]['Id']!=IMAGE:
            raise ValueError('fixed image required')
        usage=checked(['npu-smi','info'])
        if any(f'No running processes found in NPU {i}' not in usage for i in (4,5,6,7)):
            raise ValueError('leased devices occupied; retain existing work')
        root=manifest['remote_root'];package=Path(c['package'])
        storage=(package/'source/src/multimodal_jev/local_runtime_storage.py').read_text()
        checked(['python3','-c',storage+'\ncheck_paths('+repr({'root':root,'cache':config['encoded_audio_cache']})+')\n'
            +'assert not Path('+repr(root)+').exists()\n'])
        archive=base/(c['tag']+'-package.tar.gz')
        with tarfile.open(archive,'x:gz') as tar:
            for p in sorted(package.rglob('*')):
                if p.is_file():tar.add(p,arcname=str(p.relative_to(package)),recursive=False)
        write(attempt,dict(config=c,provenance=provenance,argv=argv,archive_sha256=sha(archive)))
        checked(['mkdir',root]);checked(['mkdir',root+'/exchange',root+'/cache',root+'/home'])
        subprocess.run(['scp',str(archive),c['host']+':'+root+'/package.tar.gz'],check=True,capture_output=True,timeout=55)
        checked(['tar','-xzf',root+'/package.tar.gz','-C',root])
        script='from pathlib import Path\nimport hashlib,json\nr=Path('+repr(root)+')\n'
        script+='assert hashlib.sha256((r/"PREPARED.json").read_bytes()).hexdigest()=='+repr(c['manifest_sha256'])+'\n'
        script+='m=json.loads((r/"PREPARED.json").read_text())\nassert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in m["files"].items())\n'
        checked(['python3','-c',script])
        cid=checked(argv).strip()
        if not re.fullmatch('[0-9a-f]{64}',cid):raise ValueError('invalid exact container handle')
        record=dict(container_id=cid,image=IMAGE,remote_root=root,host=c['host'],argv=argv,
            manifest_sha256=c['manifest_sha256'],config_sha256=sha(package/'config.json'),
            service_config_sha256=sha(package/'service-config.json'))
        write(handle,record)
        try:
            result=remote(['docker','start',cid])
            write(base/(c['tag']+'-start-response.json'),dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        except subprocess.TimeoutExpired:
            write(base/(c['tag']+'-start-response.json'),dict(observation_timeout=True))
        code=wait_exact(remote,cid);write(base/(c['tag']+'-wait.json'),dict(container_id=cid,exit_code=code))
        inspected=json.loads(checked(['docker','inspect',cid]))[0]
        write(base/(c['tag']+'-inspect.json'),inspected)
        result=collect(record,base/(c['tag']+'-collected'),host=c['host'])
        write(base/(c['tag']+'-collection.json'),result)
        from tools.reports.context_dual_probe_operations import check_runtime
        check_runtime(dict(argv=argv,image=IMAGE),record,inspected)
        return dict(container_id=cid,exit_code=code,status='collected_independent_resume_audit_required')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--launch',action='store_true');a=p.parse_args();c,provenance=load_config(a.config,a.local)
    if a.launch:result=launch(c,provenance)
    else:
        manifest,config=validate(c)
        result=dict(status='validated_only_no_remote_calls',argv=command(c,manifest,config))
    print(json.dumps(result,indent=2))

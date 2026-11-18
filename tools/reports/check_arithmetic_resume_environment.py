"""CPU-only preflight of a sealed recovery package against actual host assets.

Maps the package at its intended runtime path, all dependencies read-only.
Never starts the training supervisor or loads the frozen model into memory.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.prepare_context_dual_throughput_pair import sha,write
from tools.reports.launch_context_batch_probe_candidate import wait_exact

RUNNER = '''import hashlib,json,sys,time
from pathlib import Path
r=Path(sys.argv[1]);out=Path(sys.argv[2]);expected=sys.argv[3]
assert not list(Path('/dev').glob('davinci[0-9]*'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
started=time.time()
try:
 assert sha(r/'PREPARED.json')==expected
 m=json.loads((r/'PREPARED.json').read_text())
 assert all(sha(r/n)==h for n,h in m['files'].items())
 sys.path[:0]=[str(r/'source/src'),str(r/'source/tools/jev')]
 import torch
 torch.set_num_threads(1)
 from context_arithmetic_resume_runtime import validate
 from multimodal_jev.context_policy_artifacts import read_execution_bundle,verify_native_adapter
 from multimodal_jev.context_training_checkpoint import restore
 c=json.loads((r/'config.json').read_text());s=json.loads((r/'service-config.json').read_text())
 admission=validate(c)
 assert s['pins']==c['pins']
 step=c['start_update'];run=Path(c['resume_root'])
 commit=json.loads((run/f'checkpoint-{step:04d}-committed.json').read_text())
 assert s['policy']['snapshot']==str(run/f'policy-{step:04d}')
 assert s['policy']['adapter']==c['initial_adapter']==str(run/f'adapter-{step:04d}')
 assert s['policy']['policy_id']==commit['joint_policy']['policy_id']
 bundle=read_execution_bundle(s['policy']['snapshot'],expected_policy_id=s['policy']['policy_id'],
  expected_contract=c['source_contract'],expected_base_id=c['base_id'])
 adapter=verify_native_adapter(c['initial_adapter'],bundle)
 restored=[]
 for rank in range(4):
  state=torch.load(run/f'rank-{rank}/checkpoint-{step:04d}.pt',map_location='cpu',weights_only=True)
  named=[(n,torch.nn.Parameter(torch.zeros_like(v))) for n,v in state['parameters'].items()]
  opt=torch.optim.AdamW([p for _,p in named],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
  restored.append(restore(named,opt,state,rank=rank,world_size=4,completed_updates=step,contract=commit['contract']))
 result=dict(passed=True,admission=admission,native_adapter=adapter,restored=restored,
  policy_id=bundle['policy_id'],package_sha256=expected,dependency_files=len(c['pins']),
  started_unix=started,finished_unix=time.time(),torch_version=torch.__version__,
  npu_training_executed=False,full_resume_released=False,
  scope='Actual immutable host assets and native adapter tensors validated in CPU-only fixed image')
except BaseException as exc:
 result=dict(passed=False,error=repr(exc),started_unix=started,finished_unix=time.time())
 print(json.dumps(result),flush=True)
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\\n')
 raise
(out/'result.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result),flush=True)
'''


def validate(c):
    package=Path(c['package']);m=json.loads((package/'PREPARED.json').read_text())
    if sha(package/'PREPARED.json')!=c['package_sha256']:
        raise ValueError('immutable recovery package required')
    if (m['status']!='arithmetic_recovery_prefix_prepared_only' or
            m['training_dispatch_allowed'] is not False or m['finite_dispatch_allowed'] is not False):
        raise ValueError('prepared undispatched recovery prefix required')
    actual={str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file() and p.name!='PREPARED.json'}
    if any(p.is_symlink() for p in package.rglob('*')) or actual!=m['files']:
        raise ValueError('package bytes changed')
    if not re.fullmatch('[a-z0-9-]+',c['tag']) or not re.fullmatch('[a-zA-Z0-9_.-]+',c['host']):
        raise ValueError('explicit tag/host required')
    for name in ('remote_check_root','shared_root','driver_root'):
        if (not re.fullmatch('/[a-zA-Z0-9_./-]+',c[name]) or '..' in Path(c[name]).parts or
                str(Path(c[name]))!=c[name] or c[name]=='/'):
            raise ValueError('canonical isolated mount paths required')
    config=json.loads((package/'config.json').read_text())
    mounts=[(c['remote_check_root']+'/package',m['remote_root']),
        (str(Path(config['resume_origin_config']).parent),str(Path(config['resume_origin_config']).parent)),
        (c['shared_root'],c['shared_root']), (c['driver_root'],'/usr/local/Ascend/driver'),
        (str(Path(config['encoded_audio_cache']).parent),str(Path(config['encoded_audio_cache']).parent)),
        (config['audio_mirror_root'],config['audio_mirror_root'])]
    for src,dst in mounts:
        if Path(c['remote_check_root']).is_relative_to(src) or Path(src).is_relative_to(c['remote_check_root']):
            if src!=c['remote_check_root']+'/package':raise ValueError('check output overlaps inputs')
        if Path(c['remote_check_root']).is_relative_to(dst) or Path(dst).is_relative_to(c['remote_check_root']):
            raise ValueError('runtime output overlaps mounted input')
    if any(not any(Path(name).is_relative_to(dst) for _,dst in mounts) for name in config['pins']):
        raise ValueError('dependency outside read-only mounts')
    return m,mounts


def run(c,provenance):
    m,mounts=validate(c);base=Path(c['output']);check_paths({'output':base})
    base.mkdir(parents=True,exist_ok=False);root=c['remote_check_root']
    write(base/'attempt.json',dict(config=c,provenance=provenance,implementation_sha256=sha(__file__)))
    def remote(args,bounded=True):
        return subprocess.run(['ssh','-o','ConnectTimeout=10',c['host'],shlex.join(args)],
            capture_output=True,text=True,timeout=55 if bounded else None)
    def checked(args):
        r=remote(args)
        if r.returncode:raise RuntimeError(r.stderr)
        return r.stdout
    storage=Path('src/multimodal_jev/local_runtime_storage.py').read_text()
    checked(['python3','-c',storage+'\ncheck_paths({"root":'+repr(root)+'})\n'
        +'r=Path('+repr(root)+');r.mkdir(exist_ok=False)\n[(r/n).mkdir() for n in ("package","result","home","cache")]'])
    archive=base/'package.tar'
    with tarfile.open(archive,'x') as tar:
        for p in sorted(Path(c['package']).rglob('*')):
            if p.is_file():tar.add(p,arcname=str(p.relative_to(c['package'])),recursive=False)
    subprocess.run(['scp',str(archive),c['host']+':'+root+'/package.tar'],check=True,capture_output=True,timeout=55)
    checked(['tar','-xf',root+'/package.tar','-C',root+'/package'])
    (base/'check.py').write_text(RUNNER)
    subprocess.run(['scp',str(base/'check.py'),c['host']+':'+root+'/check.py'],check=True,capture_output=True,timeout=55)
    runner_hash=checked(['python3','-c',
        'import hashlib;from pathlib import Path;print(hashlib.sha256(Path('+repr(root+'/check.py')+').read_bytes()).hexdigest())']).strip()
    if runner_hash!=sha(base/'check.py'):raise ValueError('transferred CPU runner changed')
    argv=['docker','create','--pull=never','--network=none','--read-only','--name',c['tag'],
        '--cpus=2','--memory=8g','--log-driver=json-file','--log-opt','max-size=10m','--log-opt','max-file=2']
    for src,dst in mounts+[(root+'/check.py','/check.py')]:
        argv+=['--mount','type=bind,src='+src+',dst='+dst+',readonly']
    for src,dst in [(root+'/result','/result'),(root+'/home','/root'),(root+'/cache','/tmp')]:
        argv+=['--mount','type=bind,src='+src+',dst='+dst]
    for setting in ('PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','HF_HUB_OFFLINE=1',
                    'TRANSFORMERS_OFFLINE=1','TMPDIR=/tmp','ASCEND_WORK_PATH=/tmp','ASCEND_CACHE_PATH=/tmp'):
        argv+=['-e',setting]
    argv+=['--entrypoint','python',m['image'],'/check.py',m['remote_root'],'/result',c['package_sha256']]
    cid=checked(argv).strip()
    if not re.fullmatch('[0-9a-f]{64}',cid):raise ValueError('exact container identity required')
    write(base/'launch.json',dict(container_id=cid,image=m['image'],argv=argv,runner_sha256=sha(base/'check.py')))
    try:
        response=remote(['docker','start',cid]);write(base/'start.json',dict(code=response.returncode,stdout=response.stdout,stderr=response.stderr))
    except subprocess.TimeoutExpired:write(base/'start.json',dict(observation_timeout=True))
    code=wait_exact(remote,cid);write(base/'wait.json',dict(exit_code=code,container_id=cid))
    inspected=json.loads(checked(['docker','inspect',cid]))[0];write(base/'inspect.json',inspected)
    logs=checked(['docker','logs',cid]);(base/'stdout.log').write_text(logs)
    # Read result through docker cp; root-owned artifacts are not chmodded.
    import io
    raw=subprocess.check_output(['ssh',c['host'],shlex.join(['docker','cp',cid+':/result/result.json','-'])],timeout=55)
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        members=tar.getmembers()
        if len(members)!=1 or not members[0].isfile():raise ValueError('single regular result required')
        result=json.load(tar.extractfile(members[0]))
    write(base/'result.json',result)
    expected={(src,dst,False) for src,dst in mounts+[(root+'/check.py','/check.py')]}
    expected|={(root+'/'+name,dst,True) for name,dst in [('result','/result'),('home','/root'),('cache','/tmp')]}
    actual={(v['Source'],v['Destination'],v['RW']) for v in inspected['Mounts']}
    actual_env=dict(v.split('=',1) for v in inspected['Config']['Env'])
    expected_env=dict(argv[i+1].split('=',1) for i,v in enumerate(argv) if v=='-e')
    if (inspected['Id']!=cid or inspected['Image']!=m['image'] or actual!=expected or
            any(actual_env.get(k)!=v for k,v in expected_env.items()) or
            inspected['HostConfig']['Devices'] or inspected['HostConfig']['Privileged'] or
            not inspected['HostConfig']['ReadonlyRootfs'] or inspected['HostConfig']['NetworkMode']!='none' or
            inspected['Config']['Entrypoint']!=['python'] or
            inspected['Config']['Cmd']!=argv[argv.index(m['image'])+1:] or
            inspected['State']['Status']!='exited' or inspected['State']['ExitCode']!=0 or
            inspected['State']['OOMKilled'] or code!=0 or result['passed'] is not True or
            result['package_sha256']!=c['package_sha256']):
        raise ValueError('CPU runtime or preflight failed; preserve exact evidence')
    report=dict(container_id=cid,image=m['image'],state=inspected['State'],result=result,
        config_provenance=provenance,files={p.name:sha(p) for p in base.iterdir() if p.is_file()},
        scope='Actual fixed-image CPU recovery preflight; no NPU model execution or speed claim')
    write(c['report'],report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local);r=run(c,provenance)
    print(json.dumps(dict(container_id=r['container_id'],passed=r['result']['passed'])))

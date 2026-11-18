#!/usr/bin/env python3
"""Launch one fresh DP4 recovery only from accepted continuous fixture evidence."""
import argparse,hashlib,json,shlex,subprocess,sys,tarfile,time
from pathlib import Path
repository=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repository/'src'))
from multimodal_jev.resident_service_admission import IMAGE
p=argparse.ArgumentParser();p.add_argument('--algorithm',choices=['grpo_v1','rloo_v1'],default='grpo_v1')
p.add_argument('--state-dir',type=Path,default=repository/'.local/simple-final-v1')
a=p.parse_args();local=a.state_dir;base='/data/xingchao/multimodal-jev/simple-final-v1'
stem='sampled-npu-'+a.algorithm;tag=stem+'-resident-restore';remote=base+'/'+tag+'-v1'
def ssh(argv):return subprocess.check_output(['ssh','npu',shlex.join(argv)],text=True,timeout=60)
def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def put(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def copy(path,dest):subprocess.run(['scp',str(path),'npu:'+dest],check=True,capture_output=True,timeout=60)
if (local/(tag+'-created.json')).exists():raise RuntimeError('Reconcile existing exact recovery handle')
if ssh(['python3','-c','from pathlib import Path;print(Path('+repr(remote)+').exists())']).strip()=='True':
 raise RuntimeError('Existing recovery root; no overwrite')
audit_tag=stem+'-resident-train-audit';train_tag=stem+'-resident-train'
al=read(local/(audit_tag+'-launch.json'));ao=read(local/(audit_tag+'-observed.json'))
tl=read(local/(train_tag+'-launch.json'));to=read(local/(train_tag+'-observed.json'))
for launch,observed in [(al,ao),(tl,to)]:
 if (launch['image']!=IMAGE or observed['image']!=IMAGE or launch['container_id']!=observed['container_id']
         or observed['state']['Status']!='exited' or observed['state']['ExitCode']!=0):
  raise ValueError('successful exact source and independent audit required')
accepted=read(local/(audit_tag+'-verified.json'));audit_path=al['remote_root']+'/run/state-verified.json'
if (accepted.get('passed') is not True or accepted.get('updates')!=3 or accepted.get('ranks')!=4
        or accepted.get('exact_master_export') is not True or accepted.get('exact_peft_reload') is not True
        or accepted.get('exact_pre_update_recovery') is not True
        or hashlib.sha256(ssh(['cat',audit_path]).encode()).hexdigest()!=digest(local/(audit_tag+'-verified.json'))):
 raise ValueError('independently collected three-update continuous source required')
c=json.loads(ssh(['cat',tl['remote_root']+'/config.json']))
if c['physical_devices']!=[0,1,2,3]:raise ValueError('fixed DP4 topology required')
usage=ssh(['npu-smi','info'])
if not all('No running processes found in NPU '+str(i) in usage for i in c['physical_devices']):
 raise ValueError('recovery devices occupied; no preemption')
(local/(tag+'-resources.txt')).write_text(usage)
commit=json.loads(ssh(['cat',tl['remote_root']+'/run/checkpoint-0003-committed.json']))
c['continuous_resume']=dict(training_root=tl['remote_root'],completed_updates=3,algorithm=a.algorithm,
 reference_policy_id=commit['reference_policy_id'],audit=audit_path,
 audit_launch=remote+'/audit-launch.json',audit_observed=remote+'/audit-observed.json',
 training_observed=remote+'/source-observed.json')
c['pins']={k:v for k,v in c['pins'].items() if '/source/' not in k}
# Source observation must retain its audited path, rather than duplicate it at
# a new path: the independent audit file graph binds both identity and bytes.
c['continuous_resume']['training_observed']=al['remote_root']+'/training-observed.json'
for path in [audit_path,c['continuous_resume']['training_observed']]:c['pins'][path]=ssh(['sha256sum',path]).split()[0]
for suffix,source in [('audit-launch',audit_tag+'-launch'),('audit-observed',audit_tag+'-observed')]:
 c['pins'][remote+'/'+suffix+'.json']=digest(local/(source+'.json'))
with tarfile.open(local/(tag+'-source.tar.gz'),'w:gz') as archive:
 for directory in ['src','tests','tools/jev']:
  for path in sorted((repository/directory).rglob('*.py')):
   relative=str(path.relative_to(repository));archive.add(path,arcname=relative)
   c['pins'][remote+'/source/'+relative]=digest(path)
put(local/(tag+'-config.json'),c)
commands=[['python','-m','pytest','-q','-p','no:cacheprovider']+[remote+'/source/tests/'+n for n in
 ['test_simple_continuous_resume.py','test_intent_resident_replay.py','test_span_fsdp_resume.py','test_local_checkpoint.py']],
 ['torchrun','--nnodes=1','--nproc-per-node=4','--master-addr=127.0.0.1','--master-port=29755',
 remote+'/source/tools/jev/probe_simple_continuous_restore.py','--config',remote+'/config.json','--output',remote+'/run']]
runner='''import json,subprocess,time,os
from pathlib import Path
root=Path(__file__).parent
try:
 for i,cmd in enumerate(COMMANDS):
  env=dict(os.environ)
  if i==0:env['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
  with (root/('phase-'+str(i)+'.log')).open('w') as out:
   code=subprocess.call(cmd,stdout=out,stderr=subprocess.STDOUT,env=env)
  if code:raise RuntimeError('phase '+str(i)+' exit '+str(code))
 (root/'terminal.json').write_text(json.dumps(dict(exit_code=0,finished_unix=time.time())))
except BaseException as exc:
 (root/'terminal.json').write_text(json.dumps(dict(exit_code=1,error=repr(exc),finished_unix=time.time())))
 raise
'''.replace('COMMANDS',repr(commands))
(local/(tag+'-run.py')).write_text(runner)
ssh(['mkdir','-p',remote+'/source',remote+'/cache',remote+'/home'])
for name,dest in [(tag+'-source.tar.gz','source.tar.gz'),(tag+'-config.json','config.json'),(tag+'-run.py','run.py'),
 (audit_tag+'-launch.json','audit-launch.json'),(audit_tag+'-observed.json','audit-observed.json')]:copy(local/name,remote+'/'+dest)
ssh(['tar','-xzf',remote+'/source.tar.gz','-C',remote+'/source'])
old=read(local/'prefix-prepared.json')['argv']
binds=[old[i+1] for i,x in enumerate(old) if x=='-v' and ':/home/xingchao/workspace/multimodal-jev/' in old[i+1]]
argv=['docker','create','--pull=never','--network=none','--name','multimodal-jev-'+tag+'-v1',
 '--cpus=12','--memory=96g','--shm-size=8g','--log-driver=json-file','--log-opt','max-size=100m','--log-opt','max-file=3']
for device in ['davinci'+str(i) for i in c['physical_devices']]+['davinci_manager','devmm_svm','hisi_hdc']:
 argv+=['--device','/dev/'+device+':/dev/'+device]
for mount in [base+':'+base+':ro',remote+':'+remote,remote+'/source:'+remote+'/source:ro',remote+'/home:/root',
 '/mnt/cephfs/xingchao:/mnt/cephfs/xingchao:ro','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro']+binds:argv+=['-v',mount]
env=dict(PYTHONPATH=remote+'/source/src',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
 OMP_NUM_THREADS='2',HCCL_CONNECT_TIMEOUT='120',TMPDIR=remote+'/cache',XDG_CACHE_HOME=remote+'/cache',HF_HOME=remote+'/cache',
 TRANSFORMERS_CACHE=remote+'/cache',TORCHINDUCTOR_CACHE_DIR=remote+'/cache',TRITON_CACHE_DIR=remote+'/cache',
 ASCEND_WORK_PATH=remote+'/cache',ASCEND_CACHE_PATH=remote+'/cache')
for key,value in env.items():argv+=['-e',key+'='+value]
argv +=[IMAGE,'python',remote+'/run.py'];put(local/(tag+'-prepared.json'),dict(argv=argv))
cid=ssh(argv).strip();put(local/(tag+'-created.json'),dict(container_id=cid));now=time.time()
l=dict(container_id=cid,image=IMAGE,host='175',physical_devices=c['physical_devices'],remote_root=remote,
 created_unix=now,next_regular_observation_not_before=now+600,config_sha256=digest(local/(tag+'-config.json')),
 scope='Fresh DP4 continuous step3 master/Adam and full128 probability restore; no optimizer update')
put(local/(tag+'-launch.json'),l);copy(local/(tag+'-launch.json'),remote+'/launch.json')
ssh(['docker','start',cid]);l['state']=json.loads(ssh(['docker','inspect',cid]))[0]['State'];put(local/(tag+'-launch.json'),l)
print(cid,l['state']['Status'])

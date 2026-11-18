"""Two explicit phases: start isolated service, then authenticate and start DP4.

Default is engineering updates2/3; formal and recovered runs are explicit.
Never stops/restarts another container and refuses existing handles. Service
requires five free cards before initialization.
"""
import argparse,hashlib,json,re,shlex,subprocess,sys,tarfile,time
from pathlib import Path
repository=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repository/'src'))
from multimodal_jev.resident_service_admission import IMAGE,validate
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['service','trainer'],required=True)
p.add_argument('--algorithm',choices=['grpo_v1','rloo_v1'],default='grpo_v1')
p.add_argument('--formal',action='store_true',help='Full Stage2 only after released experiment and lifecycle acceptance')
p.add_argument('--fixture-resume',action='store_true',help='New service and one update from audited continuous step3')
p.add_argument('--resume-audit-tag',help='Collected stopped-prefix audit; requires --formal and a new --run-id')
p.add_argument('--run-id',help='Unique new formal segment name; leaves prior service and training untouched')
p.add_argument('--state-dir',type=Path,default=Path(__file__).resolve().parents[2]/'.local/simple-final-v1')
a=p.parse_args();local=a.state_dir;base='/data/xingchao/multimodal-jev/simple-final-v1'
if a.fixture_resume and (a.formal or a.resume_audit_tag or a.run_id):p.error('fixture resume cannot be formal or use formal resume arguments')
if a.resume_audit_tag and (not a.formal or not a.run_id):p.error('resume requires --formal and explicit new --run-id')
if a.run_id and (not a.formal or not re.fullmatch(r'[a-z0-9][a-z0-9_-]*',a.run_id)):
 p.error('run-id requires formal mode and a simple lowercase identifier')
if a.resume_audit_tag and not re.fullmatch(r'stage2-[a-z0-9_-]+',a.resume_audit_tag):p.error('explicit stage2 audit tag required')
actor_devices=[0,1,2,3];rollout_devices=[4]
fixture_stem='sampled-npu-'+a.algorithm
stem=('stage2-'+a.algorithm+('-'+a.run_id if a.run_id else '')) if a.formal else fixture_stem+('-resume' if a.fixture_resume else '');tag=stem+('-resident-service' if a.phase=='service' else '-resident-train')
service_tag=stem+'-resident-service';service_root=base+'/'+service_tag+'-v1'
remote=base+'/'+tag+'-v1';ipc=service_root+'/ipc'
def ssh(argv):return subprocess.check_output(['ssh','npu',shlex.join(argv)],text=True,timeout=60)
def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def put(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def copy(name,dest):subprocess.run(['scp',str(local/name),'npu:'+dest],check=True,capture_output=True,timeout=60)
if (local/(tag+'-created.json')).exists():raise RuntimeError('Reconcile existing exact handle; no repeat dispatch')
if ssh(['python3','-c','from pathlib import Path; print(Path('+repr(remote)+').exists())']).strip()=='True':
 raise RuntimeError('Existing remote root requires reconciliation; no overwrite')
formal_files=[];lifecycle={};experiment_path=repository/'configs/experiments/slurp-simple-sampled-rl-v1.json'
resume_source=None;resume_plan=None
fixture_source=None;fixture_recovery=None
if a.fixture_resume:
 t=fixture_stem+'-resident-train-audit';al=read(local/(t+'-launch.json'));ao=read(local/(t+'-observed.json'))
 accepted=read(local/(t+'-verified.json'));audit_path=al['remote_root']+'/run/state-verified.json'
 if (al['image']!=IMAGE or ao['image']!=IMAGE or al['container_id']!=ao['container_id']
     or ao['state']['Status']!='exited' or ao['state']['ExitCode']!=0
     or accepted.get('passed') is not True or accepted.get('updates')!=3 or accepted.get('new_updates')!=2
     or accepted.get('ranks')!=4 or len(accepted.get('checks',[]))!=2
     or any(accepted.get(k) is not True for k in ('exact_master_export','exact_peft_reload','exact_pre_update_recovery'))
     or hashlib.sha256(ssh(['cat',audit_path]).encode()).hexdigest()!=digest(local/(t+'-verified.json'))):
  raise ValueError('independently collected continuous step3 required')
 ac=json.loads(ssh(['cat',al['remote_root']+'/config.json']));source_root=ac['training_root']
 source_launch=json.loads(ssh(['cat',source_root+'/launch.json']))
 actual=json.loads(ssh(['docker','inspect',source_launch['container_id']]))[0]
 if (actual['Id']!=source_launch['container_id'] or actual['Image']!=IMAGE
     or actual['State']['Status']!='exited' or actual['State']['ExitCode']!=0):
  raise ValueError('successful exited continuous source required')
 fixture_source=json.loads(ssh(['cat',source_root+'/config.json']))
 if (fixture_source.get('sampled_probe',{}).get('algorithm')!=a.algorithm
     or fixture_source['sampled_probe'].get('fixture_initialization')!='restore_updated_rl_then_resident_loop'
     or fixture_source['physical_devices']!=actor_devices or 'stage2_initialization' in fixture_source):
  raise ValueError('same-algorithm engineering source required')
 graph_script='''# fixture_resume_evidence_graph
import hashlib,json
from pathlib import Path
p=Path(REPORT);raw=p.read_bytes()
if hashlib.sha256(raw).hexdigest()!=EXPECTED:raise ValueError('audit bytes changed')
r=json.loads(raw);files=r['files']
for name in REQUIRED:
 if name not in files:raise ValueError('missing source evidence: '+name)
for name,h in files.items():
 if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=h:raise ValueError('changed dependency: '+name)
print(json.dumps(dict(passed=True,verified_files=len(files))))
'''.replace('REPORT',repr(audit_path)).replace('EXPECTED',repr(digest(local/(t+'-verified.json')))).replace('REQUIRED',repr([source_root+'/config.json',source_root+'/launch.json',ac['training_observed']]))
 graph=json.loads(ssh(['python3','-c',graph_script]))
 if graph.get('passed') is not True or graph.get('verified_files')!=len(accepted.get('files',{})):
  raise ValueError('complete continuous evidence graph required')
 commit=json.loads(ssh(['cat',source_root+'/run/checkpoint-0003-committed.json']))
 fixture_recovery=dict(training_root=source_root,completed_updates=3,algorithm=a.algorithm,
     reference_policy_id=commit['reference_policy_id'],audit=audit_path,
     audit_launch=remote+'/resume-audit-launch.json',audit_observed=remote+'/resume-audit-observed.json',
     training_observed=ac['training_observed'])
 for suffix in ('launch','observed'):formal_files.append((local/(t+'-'+suffix+'.json'),'resume-audit-'+suffix+'.json'))
if a.formal:
 from multimodal_jev.simple_stage2_release import validate_experiment
 validate_experiment(read(experiment_path),a.algorithm)
 for role,suffix in [('continuous','resident-train-audit'),('restore','resident-restore-audit'),('resume_update','resume-resident-train-audit')]:
  t=fixture_stem+'-'+suffix;l=read(local/(t+'-launch.json'));o=read(local/(t+'-observed.json'))
  receipt=read(local/(t+'-verified.json'));report=l['remote_root']+'/run/state-verified.json'
  if (l['image']!=IMAGE or o['image']!=IMAGE or o['container_id']!=l['container_id']
      or o['state']['Status']!='exited' or o['state']['ExitCode']!=0
      or receipt.get('passed') is not True or receipt.get('updates')!=(4 if role=='resume_update' else 3)
      or hashlib.sha256(ssh(['cat',report]).encode()).hexdigest()!=digest(local/(t+'-verified.json'))):
   raise ValueError('Formal lifecycle prerequisite missing: '+role)
  audit_config=json.loads(ssh(['cat',l['remote_root']+'/config.json']))
  training_config=audit_config['training_root']+'/config.json'
  lifecycle[role]=dict(report=report,training_config=training_config,
      launch=remote+'/'+role+'-audit-launch.json',observed=remote+'/'+role+'-audit-observed.json')
  for suffix in ('launch','observed'):
   formal_files.append((local/(t+'-'+suffix+'.json'),role+'-audit-'+suffix+'.json'))
 final_launch=read(local/'stage1-selection-finalize-launch.json')
 final_obs=read(local/'stage1-selection-finalize-observed.json')
 if (final_launch['image']!=IMAGE or final_obs['image']!=IMAGE
     or final_obs['container_id']!=final_launch['container_id']
     or final_obs['state']['Status']!='exited' or final_obs['state']['ExitCode']!=0):
  raise ValueError('Successful final Stage1 selection finalizer required')
 final=json.loads(ssh(['cat',final_launch['remote_root']+'/run/result.json']))
 if (final.get('passed') is not True or final.get('updates')!=634 or final.get('recordings')!=9226
     or final.get('parity',{}).get('full_candidate_parity') is not True):
  raise ValueError('Full final Stage1 selection prerequisite missing')
 formal_files.append((experiment_path,'experiment.json'))
 if a.resume_audit_tag:
  t=a.resume_audit_tag;al=read(local/(t+'-launch.json'));ao=read(local/(t+'-observed.json'))
  accepted=read(local/(t+'-verified.json'));audit_path=al['remote_root']+'/run/state-verified.json'
  if (al['image']!=IMAGE or ao['image']!=IMAGE or al['container_id']!=ao['container_id']
      or ao['state']['Status']!='exited' or ao['state']['ExitCode']!=0
      or accepted.get('passed') is not True or accepted.get('stage')!='simple_stage2_rl'
      or accepted.get('audit_mode')!='stopped_prefix' or accepted.get('complete_training') is not False
      or type(accepted.get('updates')) is not int or not 1<=accepted['updates']<634
      or hashlib.sha256(ssh(['cat',audit_path]).encode()).hexdigest()!=digest(local/(t+'-verified.json'))):
   raise ValueError('collected formal stopped-prefix audit required')
  audit_config=json.loads(ssh(['cat',al['remote_root']+'/config.json']))
  source_root=audit_config['training_root']
  source_launch=json.loads(ssh(['cat',source_root+'/launch.json']))
  actual=json.loads(ssh(['docker','inspect',source_launch['container_id']]))[0]
  if (actual['Id']!=source_launch['container_id'] or actual['Image']!=IMAGE
      or actual['State']['Status']!='exited'):
   raise ValueError('original formal source must still be exited')
  resume_source=json.loads(ssh(['cat',source_root+'/config.json']))
  if (resume_source.get('stage2_initialization',{}).get('algorithm')!=a.algorithm
      or 'sampled_probe' in resume_source or resume_source['physical_devices']!=actor_devices):
   raise ValueError('same formal algorithm and topology required')
  from multimodal_jev.simple_resident_selection import SELECTION_UPDATES
  count=accepted['updates'];policies=accepted.get('policies',{})
  if (accepted.get('ranks')!=4 or accepted.get('new_updates')!=count
      or len(accepted.get('checks',[]))!=count
      or any(accepted.get(k) is not True for k in ('exact_master_export','exact_peft_reload','exact_pre_update_recovery'))
      or set(policies)!={str(i) for i in range(count+1)}
      or any(row.get('passed') is not True or row.get('updates')!=i
             or row.get('requests')!=128 or row.get('policy_id')!=policies[str(i)]
             for i,row in enumerate(accepted['checks'],1))
      or [v.get('updates') for v in accepted.get('selection',[])]!=[s for s in SELECTION_UPDATES if s<=count]
      or any(v.get('passed') is not True for v in accepted['selection'])
      or len(accepted.get('initial_state_checks',[]))!=4
      or any(v.get('passed') is not True for v in accepted['initial_state_checks'])):
   raise ValueError('complete independently audited recovery prefix required')
  # Rehash on the source host before creating directories or occupying a card.
  # The admitted report must bind the same source config/launch/observation.
  graph_script='''# formal_resume_evidence_graph
import hashlib,json
from pathlib import Path
report=Path(REPORT);expected=EXPECTED
if hashlib.sha256(report.read_bytes()).hexdigest()!=expected:raise ValueError('audit bytes changed')
accepted=json.loads(report.read_text());files=accepted['files']
for name in REQUIRED:
 if name not in files:raise ValueError('source missing from audit: '+name)
for name,digest in files.items():
 if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('audited dependency changed: '+name)
print(json.dumps(dict(passed=True,verified_files=len(files))))
'''.replace('REPORT',repr(audit_path)).replace('EXPECTED',repr(digest(local/(t+'-verified.json')))).replace('REQUIRED',repr([source_root+'/config.json',source_root+'/launch.json',audit_config['training_observed']]))
  graph=json.loads(ssh(['python3','-c',graph_script]))
  if graph.get('passed') is not True or graph.get('verified_files')!=len(accepted.get('files',{})):
   raise ValueError('complete remote recovery evidence graph required')
  # Preserve the original frozen experiment path/content; the mutable repo
  # config is only an additional release check, never a replacement objective.
  original_experiment=resume_source['stage2_initialization']['experiment']
  if ssh(['sha256sum',original_experiment]).split()[0]!=resume_source['pins'].get(original_experiment):
   raise ValueError('original released experiment changed')
  for suffix in ('launch','observed'):formal_files.append((local/(t+'-'+suffix+'.json'),'resume-audit-'+suffix+'.json'))
  resume_plan=dict(training_root=source_root,completed_updates=accepted['updates'],algorithm=a.algorithm,
      reference_policy_id=accepted['policies']['0'],audit=audit_path,
      audit_launch=remote+'/resume-audit-launch.json',audit_observed=remote+'/resume-audit-observed.json',
      training_observed=audit_config['training_observed'])
# Require independently collected receipts, not only successful worker exits.
prerequisites={}
for suffix,key,filename in [('switch-finalize'+('-v2' if a.algorithm=='grpo_v1' else ''),'accepted_switch','verified.json'),
 ('restore-audit','accepted_restore','state-verified.json'),('resident-batch-finalize','accepted_resident_batch','verified.json')]:
 t=fixture_stem+'-'+suffix;l=read(local/(t+'-launch.json'));o=read(local/(t+'-observed.json'));receipt=read(local/(t+'-verified.json'))
 if o['container_id']!=l['container_id'] or o['image']!=IMAGE or o['state']['Status']!='exited' or o['state']['ExitCode']!=0 or not receipt['passed']:
  raise ValueError('Independent prerequisite missing: '+t)
 path=l['remote_root']+'/run/'+filename
 if hashlib.sha256(ssh(['cat',path]).encode()).hexdigest()!=digest(local/(t+'-verified.json')):
  raise ValueError('Collected prerequisite bytes differ')
 prerequisites[key]=path
usage=ssh(['npu-smi','info']);wanted=actor_devices+rollout_devices if a.phase=='service' else actor_devices
if not all(f'No running processes found in NPU {i}' in usage for i in wanted):
 raise ValueError('Required isolated devices occupied; no preemption')
(local/(tag+'-resources.txt')).write_text(usage)
gate=(resume_source['stage2_initialization']['gate'] if resume_source else
      read(local/'stage1-selection-launch.json')['remote_root'] if a.formal else base+'/'+fixture_stem+('-parity-v2' if a.algorithm=='grpo_v1' else '-parity-v1'))
policy=json.loads(ssh(['cat',gate+'/config.json']));policy['physical_devices']=rollout_devices
if a.fixture_resume:
 # Source files are remote; derive the saved adapter identity from committed cursor.
 policy['adapter']=fixture_recovery['training_root']+'/run/loop-adapter-0003'
 prefix=policy['base_id']+':'
 if not commit['cursor']['policy_id'].startswith(prefix):raise ValueError('recovered base differs')
 policy['adapter_id']=commit['cursor']['policy_id'][len(prefix):]
# Every service/trainer consumes its own read-only source snapshot.
pins={}
with tarfile.open(local/(tag+'-source.tar.gz'),'w:gz') as archive:
 for directory in ['src','tests','tools/jev']:
  for path in sorted((repository/directory).rglob('*.py')):
   relative=str(path.relative_to(repository))
   archive.add(path,arcname=relative);pins[remote+'/source/'+relative]=digest(path)
if a.phase=='service':
 c=dict(policy=policy,socket_path='/jev-ipc/rollout.sock',exchange='/jev-ipc/exchange',
        connect_timeout_seconds=3600,request_timeout_seconds=1800,pins=pins)
 for path in [gate+'/config.json',policy['adapter']+'/adapter_config.json',policy['adapter']+'/adapter_model.safetensors']:
  c['pins'][path]=ssh(['sha256sum',path]).split()[0]
 commands=[['python','-m','pytest','-q','-p','no:cacheprovider']+[remote+'/source/tests/'+n for n in
  ['test_resident_service_admission.py','test_span_rollout_rpc.py','test_local_checkpoint.py']],
  ['python',remote+'/source/tools/jev/serve_span_rollout.py','--config',remote+'/config.json','--output',remote+'/run']]
else:
 l=read(local/(service_tag+'-launch.json'))
 if a.fixture_resume and (l.get('fixture_resume_audit')!=fixture_recovery['audit'] or l.get('resume_update')!=3):
  raise ValueError('service and trainer must bind same continuous checkpoint')
 if resume_source and (l.get('resume_audit_tag')!=a.resume_audit_tag
                       or l.get('resume_update')!=resume_plan['completed_updates']):
  raise ValueError('service and trainer must bind the same recovery audit and cursor')
 inspected=json.loads(ssh(['docker','inspect',l['container_id']]))[0]
 ready=json.loads(ssh(['cat',service_root+'/run/ready.json']))
 service_hash=ssh(['sha256sum',service_root+'/config.json']).split()[0]
 accepted=validate(l,inspected,ready,actor_devices=actor_devices,config_sha256=service_hash)
 put(local/(tag+'-service-inspect.json'),inspected);put(local/(tag+'-service-ready.json'),ready)
 put(local/(tag+'-service-admission.json'),dict(**accepted,observed_unix=time.time()))
 c=json.loads(json.dumps(fixture_source or resume_source)) if (fixture_source or resume_source) else read(local/('continuation-config.json' if a.formal else fixture_stem+'-continue-config.json'))
 c['pins']={k:v for k,v in c['pins'].items() if '/source/' not in k};c['pins'].update(pins)
 c['sampled_probe']=dict(algorithm=a.algorithm,behavior_gate=gate,reference_gate=base+'/prefix-parity-v1',
  fixture_initialization='restore_updated_rl_then_resident_loop',stop_update=3,rollout_physical_devices=rollout_devices,
  service_launch=service_root+'/launch.json',socket_path=l['socket_path'],exchange=l['exchange'],
  service_inspect=remote+'/service-inspect.json',service_ready=remote+'/service-ready.json',**prerequisites)
 if a.fixture_resume:
  c['sampled_probe'].update(fixture_initialization='restore_continuous_then_one_update',stop_update=4)
  c['continuous_resume']=fixture_recovery
  for source,dest in formal_files:c['pins'][remote+'/'+dest]=digest(source)
  for path in (fixture_recovery['audit'],fixture_recovery['training_observed']):
   c['pins'][path]=ssh(['sha256sum',path]).split()[0]
 if a.formal:
  from multimodal_jev.simple_resident_selection import SELECTION_UPDATES
  c.update(physical_devices=actor_devices,microbatch_per_rank=1,start_update=0,updates=634,total_updates=634)
  for key in ('previous_root','previous_gate','initial_adapter','accepted_adapter'):c.pop(key,None)
  plan=c.pop('sampled_probe');plan.pop('fixture_initialization');plan.pop('behavior_gate');plan.pop('reference_gate')
  plan.update(mode='final_stage1_fresh_optimizer',gate=gate,start_update=0,total_updates=634,
      stop_update=634,reset_optimizer=True,experiment=remote+'/experiment.json',lifecycle=lifecycle,
      selection_inputs=policy['inputs'],selection_updates=list(SELECTION_UPDATES))
  c['stage2_initialization']=plan
  if resume_source:
   old_plan=resume_source['stage2_initialization']
   for key in ('mode','algorithm','start_update','stop_update','total_updates','reset_optimizer',
               'gate','selection_inputs','selection_updates','experiment','lifecycle'):
    plan[key]=old_plan[key]
   c['continuous_resume']=resume_plan
   for path in (resume_plan['audit'],resume_plan['training_observed']):
    c['pins'][path]=ssh(['sha256sum',path]).split()[0]
  for source,dest in formal_files:c['pins'][remote+'/'+dest]=digest(source)
  for entry in lifecycle.values():
   for key in ('report','training_config'):c['pins'][entry[key]]=ssh(['sha256sum',entry[key]]).split()[0]
  for path in [gate+'/config.json',gate+'/launch.json',gate+'/observed-final.json',policy['inputs']]:
   c['pins'][path]=ssh(['sha256sum',path]).split()[0]
 for path in [service_root+'/launch.json',service_root+'/config.json',*prerequisites.values()]:
  c['pins'][path]=ssh(['sha256sum',path]).split()[0]
 for suffix in ['service-inspect','service-ready','service-admission']:
  c['pins'][remote+'/'+suffix+'.json']=digest(local/(tag+'-'+suffix+'.json'))
 commands=[['python','-m','pytest','-q','-p','no:cacheprovider']+[remote+'/source/tests/'+n for n in
  ['test_resident_service_admission.py','test_simple_resident_acceptance.py','test_simple_resident_batch.py',
   'test_simple_sampled_rollout.py','test_span_training_session.py','test_span_fsdp_resume.py','test_local_checkpoint.py']],
  ['torchrun','--nnodes=1','--nproc-per-node=4','--master-addr=127.0.0.1','--master-port=29745',
   remote+'/source/tools/jev/probe_simple_resident_train.py','--config',remote+'/config.json','--output',remote+'/run']]
 if a.formal:
  commands[0]+=[remote+'/source/tests/'+n for n in ('test_simple_stage2_release.py','test_simple_stage2_initialization.py')]
  commands[1][commands[1].index(remote+'/source/tools/jev/probe_simple_resident_train.py')]=remote+'/source/tools/jev/train_simple_stage2_npu.py'
  # Authenticate the immutable release graph before torchrun constructs actors.
  commands.insert(1,['python','-c','import json;from multimodal_jev.simple_stage2_release import authenticate;authenticate(json.load(open('+repr(remote+'/config.json')+')))'])
  if resume_source:
   commands.insert(2,['python','-c','import sys,json;sys.path.insert(0,'+repr(remote+'/source/tools/jev')+');from multimodal_jev.simple_stage2_resume import authenticate;from verify_slurp_simple_parity import verify;authenticate(json.load(open('+repr(remote+'/config.json')+')),verify_parity=verify)'])
 if a.fixture_resume:
  commands[0]+=[remote+'/source/tests/test_simple_fixture_resume.py']
  commands[1][commands[1].index(remote+'/source/tools/jev/probe_simple_resident_train.py')]=remote+'/source/tools/jev/probe_simple_continuous_resume_update.py'
put(local/(tag+'-config.json'),c)
runner='''import subprocess,json,time,os
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
if a.phase=='service':ssh(['mkdir','-p',ipc+'/exchange'])
for name,dest in [(tag+'-source.tar.gz','source.tar.gz'),(tag+'-config.json','config.json'),(tag+'-run.py','run.py')]:copy(name,remote+'/'+dest)
for source,dest in formal_files:
 subprocess.run(['scp',str(source),'npu:'+remote+'/'+dest],check=True,capture_output=True,timeout=60)
if a.phase=='trainer':
 for suffix in ['service-inspect','service-ready','service-admission']:copy(tag+'-'+suffix+'.json',remote+'/'+suffix+'.json')
ssh(['tar','-xzf',remote+'/source.tar.gz','-C',remote+'/source'])
old=read(local/'prefix-prepared.json')['argv']
historical=[old[i+1] for i,x in enumerate(old) if x=='-v' and ':/home/xingchao/workspace/multimodal-jev/' in old[i+1]]
driver=[old[i+1] for i,x in enumerate(old) if x=='-v' and any(s in old[i+1] for s in ['/usr/local/Ascend','/etc/ascend','/usr/local/bin/npu-smi'])]
argv=['docker','create','--pull=never','--network=none','--name','multimodal-jev-'+tag+'-v1',
 '--cpus','8' if a.phase=='service' else '12','--memory','48g' if a.phase=='service' else '96g','--shm-size=8g',
 '--log-driver=json-file','--log-opt','max-size=100m','--log-opt','max-file=3']
devices=rollout_devices if a.phase=='service' else actor_devices
for name in ['davinci'+str(i) for i in devices]+['davinci_manager','devmm_svm','hisi_hdc']:
 argv+=['--device','/dev/'+name+':/dev/'+name]
for mount in [base+':'+base+':ro',remote+':'+remote,remote+'/source:'+remote+'/source:ro',remote+'/home:/root',
 remote+'/cache:/jev-tmp',ipc+':/jev-ipc','/mnt/cephfs/xingchao:/mnt/cephfs/xingchao:ro']+historical+driver:
 argv+=['-v',mount]
env=dict(PYTHONPATH=remote+'/source/src',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
 OMP_NUM_THREADS='2',HCCL_CONNECT_TIMEOUT='120',VLLM_WORKER_MULTIPROC_METHOD='spawn',TMPDIR='/jev-tmp',
 VLLM_RPC_BASE_PATH='/jev-tmp',XDG_CACHE_HOME=remote+'/cache',HF_HOME=remote+'/cache',TRANSFORMERS_CACHE=remote+'/cache',
 TORCHINDUCTOR_CACHE_DIR=remote+'/cache',TRITON_CACHE_DIR=remote+'/cache',ASCEND_WORK_PATH=remote+'/cache',ASCEND_CACHE_PATH=remote+'/cache')
for k,v in env.items():argv+=['-e',k+'='+v]
argv +=[IMAGE,'python',remote+'/run.py'];put(local/(tag+'-prepared.json'),dict(argv=argv))
cid=ssh(argv).strip();put(local/(tag+'-created.json'),dict(container_id=cid));now=time.time()
l=dict(container_id=cid,image=IMAGE,host='175',physical_devices=devices,remote_root=remote,created_unix=now,
 next_regular_observation_not_before=now+600,config_sha256=digest(local/(tag+'-config.json')),
 scope='Isolated resident TP1 service' if a.phase=='service' else 'Two continuous full128 fixture updates with resident DP4 and separate TP1; not formal Stage2')
if a.formal:l['scope']='Formal full40513 Stage2 '+a.phase+'; final Stage1, same runtime, fixed reference and full selection schedule'
if resume_source:l.update(resume_audit_tag=a.resume_audit_tag,resume_update=resume_plan['completed_updates'])
if a.fixture_resume:
 l.update(fixture_resume_audit=fixture_recovery['audit'],resume_update=3,
          scope='Engineering continuous checkpoint3 fresh-process restore and scheduled update4; '+a.phase)
if a.phase=='service':l.update(socket_path=c['socket_path'],exchange=c['exchange'],ipc_host=ipc,base_id=policy['base_id'],runtime_profile=policy['runtime_profile'])
put(local/(tag+'-launch.json'),l);copy(tag+'-launch.json',remote+'/launch.json')
ssh(['docker','start',cid]);l['state']=json.loads(ssh(['docker','inspect',cid]))[0]['State'];put(local/(tag+'-launch.json'),l)
print(cid,l['state']['Status'])

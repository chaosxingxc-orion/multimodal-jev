#!/usr/bin/env python3
"""Prepare or launch full fixed-initial/final evaluation after actual chain audit."""
import json,shlex,shutil,subprocess
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.policy_identity import checkpoint_identity
import argparse
from verify_span_transfer_chain import verify as verify_chain
p=argparse.ArgumentParser(description='Authenticate completed trajectory and prepare/launch its fixed evaluation')
p.add_argument('--training',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
p.add_argument('--device',type=int,choices=(4,5,6,7),required=True);p.add_argument('--launch',action='store_true')
a=p.parse_args();repo=Path(__file__).resolve().parents[2]
parent=Path('/mnt/cephfs/xingchao/experiments/multimodal-jev');training=a.training.resolve();root=a.root.resolve()
if training.parent!=parent or root.parent!=parent or root.exists():raise ValueError('existing training and exclusive shared output root required')
verified=verify_chain(training)
study=json.loads((training/'protocol.json').read_text());count=study['updates_per_branch']
if not verified['passed'] or verified['completed_updates']!=count:raise ValueError('verified terminal trajectory required')
root.mkdir(exist_ok=False);source=root/'source'
atomic_json(root/'training-verified.json',verified)
for folder in ('src','tools','configs/tasks/slurp/temporal'):
 shutil.copytree(repo/folder,source/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copyfile(training/'protocol.json',root/'study.json');study=json.loads((root/'study.json').read_text())
initial=Path(study['initializer']['root'])/'run/adapter';final=training/'run'/f'update-{count:04d}'/'run/policy-1/adapter'
model=Path(json.loads((training/'config.json').read_text())['model']);base=checkpoint_identity(model)[0]
protocol=dict(schema='jev-transfer-frozen-evaluation-v1',study_sha256=file_hash(root/'study.json'),optimizer_updates=0,training_admitted=False,policy_ids=[base+':'+checkpoint_identity(a)[0] for a in (initial,final)],training_root=str(training),terminal_update=count,scope='Fixed initial and terminal policies; all learning and reserved-source rules; no checkpoint selection')
atomic_json(root/'evaluation-protocol.json',protocol)
manifest=dict(parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files={str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()})
atomic_json(root/'source-manifest.json',manifest)
stages=[]
for role in ('learning','selection'):
 for policy,adapter in [('initial',initial),('final',final)]:
  stage=root/(policy+'-'+role);stage.mkdir();release=Path(study[role+'_release'])
  rows=[json.loads(s) for s in (release/(role+'.jsonl')).read_text().splitlines()]
  config=dict(model=str(model),adapter=str(adapter),base_id=base,adapter_id=checkpoint_identity(adapter)[0],release=str(release),population='transfer-full-evaluation-v1',evaluation_role=role,evaluation_protocol=str(root/'evaluation-protocol.json'),transfer_protocol=str(root/'study.json'),physical_devices=[a.device],runtime_profile=study['runtime_profile'],request_ids=[r['id'] for r in rows],pins={str(source/p):d for p,d in manifest['files'].items()})
  for p in [root/'study.json',root/'evaluation-protocol.json',*adapter.iterdir()]:
   if p.is_file():config['pins'][str(p)]=file_hash(p)
  atomic_json(stage/'config.json',config);atomic_json(stage/'source-manifest.json',manifest);stages.append(str(stage))
controller=source/'tools/jev/run_span_transfer_evaluation.py'
plan=dict(schema='jev-transfer-evaluation-plan-v1',stages=[Path(s).name for s in stages],
 pins={str(root/n):file_hash(root/n) for n in ('study.json','evaluation-protocol.json','source-manifest.json','training-verified.json')})
for stage in map(Path,stages):
 for name in ('config.json','source-manifest.json'):plan['pins'][str(stage/name)]=file_hash(stage/name)
atomic_json(root/'plan.json',plan)
image=json.loads((training/'launch.json').read_text())['image']
argv=['docker','run','-d','--pull=never','--network=none','--name','multimodal-jev-'+root.name,'--cpus=8','--memory=64g','--shm-size=4g']
for dev in (f'davinci{a.device}','davinci_manager','devmm_svm','hisi_hdc'):argv+=['--device',f'/dev/{dev}:/dev/{dev}']
for env in ['PYTHONDONTWRITEBYTECODE=1','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','TOKENIZERS_PARALLELISM=false','OMP_NUM_THREADS=4','PYTHONPATH='+str(source/'src')]:argv+=['-e',env]
for path,mode in [('/usr/local/Ascend/driver','ro'),('/mnt/cephfs/xingchao/models','ro'),('/mnt/cephfs/xingchao/datasets','ro'),(str(parent),'ro'),(str(root),'rw'),(str(source),'ro')]:argv+=['-v',f'{path}:{path}:{mode}']
argv+=['-v',f'{source}/src:{repo}/src:ro','-v',f'{source}/tools:{repo}/tools:ro','-v',f'{source}/configs/tasks/slurp/temporal/tasks-v1.json:{repo}/configs/tasks/slurp/temporal/tasks-v1.json:ro']
argv += [image,'timeout','--signal=TERM','--kill-after=30s','14400s','python3','-B',str(controller),'--root',str(root)]
for stage in map(Path,stages):
 atomic_json(stage/'launch.json',dict(root=str(stage),argv=argv,image=image,physical_devices=[a.device],execution='stage in shared evaluation container',config_sha256=file_hash(stage/'config.json'),source_manifest_sha256=file_hash(stage/'source-manifest.json'),controller_sha256=file_hash(controller)))
for p in (root,*root.rglob('*')):p.chmod(0o777 if p.is_dir() else 0o644)
receipt=dict(root=str(root),argv=argv,image=image,physical_devices=[a.device],
 controller_sha256=file_hash(controller),evaluation_protocol_sha256=file_hash(root/'evaluation-protocol.json'),
 plan_sha256=file_hash(root/'plan.json'),training_verification_sha256=file_hash(root/'training-verified.json'))
atomic_json(root/'prepared-launch.json',receipt)
if a.launch:
 r=subprocess.run(['ssh','npu',shlex.join(argv)],capture_output=True,text=True,timeout=60)
 receipt.update(launch_exit_code=r.returncode,container_id=r.stdout.strip(),stderr=r.stderr)
 atomic_json(root/'launch.json',receipt);atomic_json(repo/'docs/engineering'/(root.name+'-launch.json'),receipt)
 print(json.dumps({k:receipt[k] for k in ('root','container_id','launch_exit_code','stderr')}));r.check_returncode()
else:
 print(json.dumps(dict(root=str(root),prepared=True,launched=False,argv=argv),indent=2))

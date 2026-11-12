#!/usr/bin/env python3
"""Launch a prespecified branch using a complete current source snapshot.

Never copy a historical launch argv or partially patch an old snapshot.
Physical NPU availability must be checked before invoking --launch.
"""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.span_transfer_runtime import branch_protocol, step_rows


def prepare(args):
    repo = Path(__file__).resolve().parents[2]
    parent = Path('/mnt/cephfs/xingchao/experiments/multimodal-jev')
    root = args.root.resolve()
    if root.parent != parent or root.exists():
        raise ValueError('exclusive shared experiment root required')
    study = repo/'docs/engineering/span-rule-transfer-protocol-v1.json'
    protocol = json.loads(study.read_text())
    config = dict(model='/mnt/cephfs/xingchao/models/Qwen__Qwen3-ASR-0.6B-hf',
        release=protocol['learning_release'], transfer_protocol=str(study), seed=args.seed,
        method=args.method, arm=args.arm, auxiliary_weight=args.auxiliary_weight,
        runtime_profile=protocol['runtime_profile'], lr=protocol['optimizer']['lr'],
        samples_per_request=protocol['joint_grpo_samples_per_request'],
        num_updates=protocol['updates_per_branch'], start_index=0,
        physical_devices=[args.device], pins={str(study):file_hash(study)})
    branch_protocol(config)
    for index in range(config['num_updates']):
        step_rows({**config,'batch_index':index,
                   'request_ids':protocol['schedules'][str(args.seed)][index]['arms'][args.arm]})
    root.mkdir(); source=root/'source'
    for folder in ('src','tools','configs/tasks/slurp/temporal'):
        shutil.copytree(repo/folder,source/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile(study,root/'protocol.json')
    manifest=dict(parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        files={str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()})
    # Every snapshot file must equal the current reviewed file, not an old run.
    for name,digest in manifest['files'].items():
        if file_hash(repo/name)!=digest:raise ValueError('source changed while snapshotting')
        if name.endswith('.py'):compile((source/name).read_bytes(),name,'exec')
    atomic_json(root/'source-manifest.json',manifest)
    config['transfer_protocol']=str(root/'protocol.json')
    config['pins']={str(source/p):digest for p,digest in manifest['files'].items()}
    config['pins'][str(root/'protocol.json')]=file_hash(root/'protocol.json')
    atomic_json(root/'config.json',config)
    image='sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'
    argv=['docker','run','-d','--pull=never','--network=none','--name','multimodal-jev-'+root.name,
          '--cpus=8','--memory=64g','--shm-size=4g']
    for dev in (f'davinci{args.device}','davinci_manager','devmm_svm','hisi_hdc'):
        argv+=['--device',f'/dev/{dev}:/dev/{dev}']
    for env in ('PYTHONDONTWRITEBYTECODE=1','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',
                'TOKENIZERS_PARALLELISM=false','OMP_NUM_THREADS=4','VLLM_WORKER_MULTIPROC_METHOD=spawn',
                'PYTHONPATH='+str(source/'src')):
        argv+=['-e',env]
    for path,mode in [('/usr/local/Ascend/driver','ro'),('/mnt/cephfs/xingchao/models','ro'),
                      ('/mnt/cephfs/xingchao/datasets','ro'),(str(parent),'ro'),(str(root),'rw'),(str(source),'ro')]:
        argv+=['-v',f'{path}:{path}:{mode}']
    for rel in ('src','tools','configs/tasks/slurp/temporal/tasks-v1.json'):
        argv+=['-v',f'{source/rel}:{repo/rel}:ro']
    controller=source/'tools/jev/run_span_transfer_chain.py'
    argv += [image,'timeout','--signal=TERM','--kill-after=30s','14400s','python3','-B',str(controller),
             '--config',str(root/'config.json'),'--output',str(root/'run')]
    receipt=dict(root=str(root),argv=argv,image=image,physical_devices=[args.device],
        config_sha256=file_hash(root/'config.json'),source_manifest_sha256=file_hash(root/'source-manifest.json'),
        protocol_sha256=file_hash(root/'protocol.json'),controller_sha256=file_hash(controller))
    for p in (root,*root.rglob('*')):p.chmod(0o777 if p.is_dir() else 0o644)
    atomic_json(root/'prepared-launch.json',receipt)
    if args.launch:
        result=subprocess.run(['ssh','-o','ControlMaster=no','-o','ControlPath=none','npu',shlex.join(argv)],
                              capture_output=True,text=True,timeout=60)
        receipt.update(launch_exit_code=result.returncode,container_id=result.stdout.strip(),stderr=result.stderr)
        atomic_json(root/'launch.json',receipt)
        atomic_json(repo/'docs/engineering'/(root.name+'-launch.json'),receipt)
        result.check_returncode()
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--method',choices=('sft','rl','conditional'),required=True)
    p.add_argument('--arm',choices=('final_only','structured'),required=True)
    p.add_argument('--auxiliary-weight',type=float,required=True)
    p.add_argument('--device',type=int,choices=(4,5,6,7),required=True)
    p.add_argument('--launch',action='store_true');a=p.parse_args();r=prepare(a)
    print(json.dumps({k:v for k,v in r.items() if k!='argv'},indent=2))

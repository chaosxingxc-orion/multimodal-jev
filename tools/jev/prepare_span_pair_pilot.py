#!/usr/bin/env python3
"""Prepare an exclusive paired one-step engineering probe; never launch it."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_pair_pilot import pilot_rows


def prepare(root):
    repo=Path(__file__).resolve().parents[2]
    parent=Path('/mnt/cephfs/xingchao/experiments/multimodal-jev')
    root=root.resolve()
    if root.parent!=parent or root.exists():raise ValueError('exclusive shared experiment root required')
    source_step=parent/'span-transfer-conditional-s17-a2-v3/run/update-0001/config.json'
    batch=json.loads(source_step.read_text())
    cpu=repo/'docs/engineering/span-pair-verl-cpu-v2-verified.json'
    verified=json.loads(cpu.read_text())
    if verified['passed'] is not True or verified['cases']!=15:raise ValueError('CPU verl verification required')
    for path,digest in verified['files'].items():
        if file_hash(path)!=digest:raise ValueError('CPU verification changed')
    for name in ('span_pair_credit.py','span_pair_verl.py'):
        if file_hash(repo/'src/multimodal_jev'/name)!=file_hash(Path(verified['root'])/'source/src/multimodal_jev'/name):
            raise ValueError('loss differs from CPU verified source')
    root.mkdir();source=root/'source'
    for folder in ('src','tools','configs/tasks/slurp/temporal'):
        shutil.copytree(repo/folder,source/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    manifest=dict(parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        files={str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()})
    for name,digest in manifest['files'].items():
        if file_hash(repo/name)!=digest:raise ValueError('source changed during preparation')
        if name.endswith('.py'):compile((source/name).read_bytes(),name,'exec')
    atomic_json(root/'source-manifest.json',manifest)
    protocol=dict(schema='jev-pair-engineering-pilot-v1',algorithm='exact-complement-pair-marginal-ppo-v1',
        pair_weights=[0.,.5],num_updates=1,auxiliary_weight=0.,clip_ratio=.2,optimizer_start='fresh',
        lr=1e-4,seed=17,arm='structured',batch_index=0,source_step=str(source_step),
        initial_adapter_id=batch['adapter_id'],base_id=batch['base_id'],runtime_admission=batch['runtime_admission'],
        pins={str(p):file_hash(p) for p in [source_step,cpu,*map(Path,verified['files'])]},
        scope='Independent engineering probe only. Fixed first learning batch, beta0 control and beta0.5 candidate. '
              'No selection/test optimization, no scientific gain claim, no continuation without separate full gates.')
    atomic_json(root/'protocol.json',protocol)
    configs=[]
    for name,beta in [('pair0',0.),('pair05',.5)]:
        config=dict(protocol=str(root/'protocol.json'),pair_weight=beta,physical_devices=[5],
            pins={str(source/name):digest for name,digest in manifest['files'].items()})
        config['pins'][str(root/'protocol.json')]=file_hash(root/'protocol.json')
        _,_,rows,pairs=pilot_rows(config)
        atomic_json(root/(name+'.json'),config);configs.append(str(root/(name+'.json')))
    for path in [root,*root.rglob('*')]:path.chmod(0o777 if path.is_dir() else 0o644)
    receipt=dict(root=str(root),prepared=True,launched=False,configs=configs,
        protocol_sha256=file_hash(root/'protocol.json'),source_manifest_sha256=file_hash(root/'source-manifest.json'),
        image=verified['image'],requests_per_branch=len(rows),pairs_per_branch=len(pairs),
        controller=str(source/'tools/jev/run_span_pair_pilot.py'),
        launch_blockers=['Wait for existing experiments; verify physical card availability.',
                        'Create authenticated launch receipt and isolated pinned-image container for controller.'],
        scope='Prepared full-source training entry and fixed one-step configs only; no NPU execution or accepted checkpoint.')
    atomic_json(root/'prepared.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=prepare(a.root);atomic_json(a.output,r);print(json.dumps(r,indent=2))

#!/usr/bin/env python3
"""Freeze the first D binding update and independent suite without dispatch."""
import argparse
import json
from pathlib import Path
import shutil
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.container_pins import validate_bind_pins
from multimodal_jev.intent_binding_resume import SCHEMA,validate_config,expected_selection
from multimodal_jev.intent_binding_inputs import load
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.span_candidate_intervention import VARIANTS
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read


def prepare(root,training_template,suite_template,gate,release):
    root,training_template,suite_template,gate,release=map(Path,(root,training_template,suite_template,gate,release))
    if root.exists():raise FileExistsError('immutable binding plan exists')
    repo=Path(__file__).resolve().parents[2]
    reference=read(training_template/'config-template.json');gc=read(gate/'config.json')
    gp=read(gate/'prepared.json');manifest=read(release/'MANIFEST.json')
    if (reference['step']!=0 or reference['arm']!='D' or reference['global_requests']!=8
            or gc['population']!=dict(recordings=89,requests=712)
            or gc['adapter']!=reference['initial_adapter'] or gc['model']!=reference['model_path']
            or gp['physical_devices']!=[4] or file_hash(gate/'config.json')!=gp['config_sha256']):
        raise ValueError('original D initialization and complete binding gate required')
    check_pins(gc['pins']);rows,_=load(release,file_hash(release/'MANIFEST.json'),'D')
    cfg={k:v for k,v in reference.items() if k not in ('pins','pilot','manifest_sha256')}
    cfg.update(schema=SCHEMA,binding_release=str(release),binding_manifest_sha256=file_hash(release/'MANIFEST.json'),
        source_batch=8,views_per_source=4,global_requests=32,world_size=4,variants=list(VARIANTS),
        acceptance_root=str(gate),image=gp['image'],pins={})
    validate_config(cfg);selected,selection=expected_selection(cfg)
    base_id,model_files=checkpoint_identity(cfg['model_path']);aid,adapter_files=checkpoint_identity(cfg['initial_adapter'])
    if base_id!=gc['base_id'] or aid!=gc['adapter_id']:raise ValueError('actual initial policy differs')
    source_pilot=read(Path(manifest['source_pilot'])/'MANIFEST.json')
    ledger=read(Path(source_pilot['release'])/'review-ledger.json')
    external={**ledger['evidence_files'],**manifest['implementation_sha256']}
    external.update({str(release/'MANIFEST.json'):file_hash(release/'MANIFEST.json')})
    external.update({str(release/p):h for p,h in manifest['files'].items()})
    external.update({str(Path(cfg['model_path'])/p):h for p,h in model_files.items()})
    external.update({str(Path(cfg['initial_adapter'])/p):h for p,h in adapter_files.items()})
    external.update({str(gate/p):file_hash(gate/p) for p in ('config.json','prepared.json','source-manifest.json')})
    external[gc['inputs']]=file_hash(gc['inputs'])
    check_pins(external)
    root.mkdir();train=root/'training';suite=root/'suite'
    protocol=dict(schema='intent-binding-first-update-protocol-v1',arm='D',updates=1,seed=cfg['seed'],lr=cfg['lr'],
        initial_policy_id=base_id+':'+aid,binding_manifest_sha256=cfg['binding_manifest_sha256'],
        selection=selection,source_uses=8,view_uses=32,reference_training=str(training_template),
        expected_backend_requests=712,scope='First binding engineering update; not a matched efficacy curve or target-domain adaptation.',
        execution_order='After all existing queued work including beta1zero continuation; actual input gate and card ownership required.')
    atomic_json(root/'protocol.json',protocol);external[str(root/'protocol.json')]=file_hash(root/'protocol.json')
    for out,template in ((train,training_template),(suite,suite_template)):
        out.mkdir()
        for folder in ('src','tools'):
            shutil.copytree(repo/folder,out/'source'/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        snapshot={str(p.relative_to(out/'source')):file_hash(p) for p in (out/'source').rglob('*') if p.is_file()}
        check_pins({str(repo/p):h for p,h in snapshot.items()})
        atomic_json(out/'source-manifest.json',snapshot)
        oldplan=read(template/'prepared.json')
        if oldplan['image']!=cfg['image']:raise ValueError('runtime image differs')
        argv=[a.replace(str(template),str(out)) for a in oldplan['create_argv']]
        argv[argv.index('--name')+1]='multimodal-jev-'+root.name+'-'+out.name
        # Replace historical evidence mounts with exact copied source evidence.
        new=[];i=0
        while i<len(argv):
            if argv[i]=='-v' and argv[i+1].split(':')[1].startswith('/home/'):
                i+=2;continue
            new.append(argv[i]);i+=1
        argv=new;extra=[]
        for index,(path,h) in enumerate(external.items()):
            if not path.startswith('/home/'):continue
            dest=out/'evidence'/f'{index}-{Path(path).name}';dest.parent.mkdir(exist_ok=True);shutil.copy2(path,dest)
            if file_hash(dest)!=h:raise ValueError('external evidence changed')
            extra+=['-v',f'{dest}:{path}:ro']
        image_index=argv.index(cfg['image']);argv[image_index:image_index]=extra
        pins={**external,**{str(out/'source'/p):h for p,h in snapshot.items()}}
        if out==train:
            current=dict(cfg,pins=pins)
            argv=[a.replace('train_intent_finite_npu.py','train_intent_binding_npu.py') for a in argv]
            argv=[('3600s' if a=='1800s' else a) for a in argv]
            pending=[str(gate/p) for p in ('launch.json','run/READY.json','run/terminal.json',
                'run/hf/result.json','run/hf/terminal.json','run/vllm/result.json','run/vllm/terminal.json')]
        else:
            current={k:v for k,v in gc.items() if k!='pins'}
            current.update(training_root=str(train),adapter=str(train/'run/adapter'),adapter_id=None,pins=pins)
            argv=[a.replace('run_intent_finite_acceptance.py','run_intent_binding_acceptance.py') for a in argv]
            argv=[('14400s' if a=='3600s' else a) for a in argv];pending=[]
        atomic_json(out/'config-template.json',current)
        plan=dict(root=str(out),image=cfg['image'],physical_devices=[4,5,6,7] if out==train else [4],
            create_argv=argv,source_manifest_sha256=file_hash(out/'source-manifest.json'),
            config_template_sha256=file_hash(out/'config-template.json'),pending_pins=pending,status='prepared_not_dispatched')
        if out==suite:plan['training_root']=str(train)
        atomic_json(out/'known-bind-pins.json',validate_bind_pins(argv,pins));atomic_json(out/'prepared.json',plan)
    receipt=dict(root=str(root),training=str(train),suite=str(suite),gate=str(gate),
        policy_id=base_id+':'+aid,protocol_sha256=file_hash(root/'protocol.json'),dispatch_allowed=False,
        scope='Immutable first-update plan only. Full real binding policy gate and existing queue predecessors must finish before freeze/dispatch.')
    atomic_json(root/'prepared.json',receipt)
    for p in [root,*root.rglob('*')]:p.chmod(0o777 if p.is_dir() else 0o644)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','training-template','suite-template','gate','release'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.root,a.training_template,a.suite_template,a.gate,a.release),indent=2))

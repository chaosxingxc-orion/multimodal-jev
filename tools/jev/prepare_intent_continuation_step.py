#!/usr/bin/env python3
"""Prepare an immutable continuation pair without dispatch or future-result pins."""
import argparse
import json
from pathlib import Path
import shutil
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.container_pins import validate_bind_pins
from multimodal_jev.intent_resume import validate_lineage


def prepare(training,suite,previous_training,previous_suite,training_template,suite_template):
    training,suite,previous_training,previous_suite,training_template,suite_template=map(Path,
        (training,suite,previous_training,previous_suite,training_template,suite_template))
    if training.exists() or suite.exists() or training==suite:
        raise ValueError('two exclusive new output roots required')
    read=lambda p:json.loads(p.read_text());repo=Path(__file__).resolve().parents[2]
    previous=read(previous_training/'config-template.json')
    outputs=[]
    for root,template in ((training,training_template),(suite,suite_template)):
        root.mkdir(parents=True)
        for folder in ('src','tools'):
            shutil.copytree(repo/folder,root/'source'/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        manifest={str(p.relative_to(root/'source')):file_hash(p) for p in (root/'source').rglob('*') if p.is_file()}
        for name,h in manifest.items():
            if file_hash(repo/name)!=h:raise ValueError('source changed during snapshot')
            if name.endswith('.py'):compile((root/'source'/name).read_bytes(),name,'exec')
        atomic_json(root/'source-manifest.json',manifest)
        cfg=read(template/'config-template.json');plan=read(template/'prepared.json')
        if (template/'evidence').exists():shutil.copytree(template/'evidence',root/'evidence')
        cfg['pins']={p.replace(str(template),str(root)):h for p,h in cfg['pins'].items()
            if not p.startswith(str(template/'source')+'/')}
        cfg['pins'].update({str(root/'source'/p):h for p,h in manifest.items()})
        if root==training:
            cfg.update(step=previous['step']+1,previous_root=str(previous_training),previous_suite=str(previous_suite),
                acceptance_root=str(previous_suite),initial_adapter=str(previous_training/'run/adapter'))
            validate_lineage(previous,cfg)
        else:
            cfg.update(training_root=str(training),adapter=str(training/'run/adapter'),adapter_id=None)
        argv=[a.replace(str(template),str(root)) for a in plan['create_argv']]
        argv[argv.index('--name')+1]='multimodal-jev-'+root.name
        atomic_json(root/'config-template.json',cfg)
        plan.update(root=str(root),create_argv=argv,status='prepared_not_dispatched',
            source_manifest_sha256=file_hash(root/'source-manifest.json'),config_template_sha256=file_hash(root/'config-template.json'),
            scope='Fixed continuation pair; predecessor results will be authenticated and pinned only after actual acceptance.')
        if root==suite:plan['training_root']=str(training)
        atomic_json(root/'known-bind-pins.json',validate_bind_pins(argv,cfg['pins']))
        atomic_json(root/'prepared.json',plan);outputs.append(plan)
        for p in [root,*root.rglob('*')]:p.chmod(0o777 if p.is_dir() else 0o644)
    return dict(training=outputs[0],suite=outputs[1],previous_updates=previous['step']+1,updates=previous['step']+2)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('training','suite','previous-training','previous-suite','training-template','suite-template'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(prepare(**vars(a)),indent=2))

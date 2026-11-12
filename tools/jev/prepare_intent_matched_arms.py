#!/usr/bin/env python3
"""Prepare Q/P first updates from D's fixed initialization; never dispatch."""
import argparse
import json
from pathlib import Path
import shutil
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.container_pins import validate_bind_pins
from multimodal_jev.slurp_intent_pilot import load_pilot
from multimodal_jev.intent_actor_inputs import schedule


def audit_arms(populations,seed,updates):
    if set(populations)!={'D','Q','P'}:raise ValueError('all matched arms required')
    ids=[r['id'] for r in populations['D']]
    if any([r['id'] for r in populations[a]]!=ids for a in ('Q','P')):raise ValueError('arm populations differ')
    for d,q,p in zip(*(populations[a] for a in ('D','Q','P')),strict=True):
        if q['request']!=p['request']:raise ValueError('Q/P inputs must be identical')
        if any(r['request'].questions[-1]!=d['request'].questions[-1] or r['request'].audio!=d['request'].audio
                or r['request'].requirements!=d['request'].requirements for r in (q,p)):
            raise ValueError('matched final knowledge and audio required')
        if any(r['target']['targets'][-1]!=d['target']['targets'][-1]
                or r['target']['request_weight']!=d['target']['request_weight']
                or r['target']['source_group']!=d['target']['source_group'] for r in (q,p)):
            raise ValueError('final supervision or outer weights differ')
        if ([d['target']['span_weights'],q['target']['span_weights'],p['target']['span_weights']]
                !=[[1.],[0.,0.,1.],[.1,.1,1.]]):raise ValueError('fixed D/Q/P credit required')
    batches={a:[[r['id'] for r in schedule(populations[a],seed,n,8)] for n in range(updates)] for a in ('D','Q','P')}
    if batches['D']!=batches['Q'] or batches['Q']!=batches['P']:raise ValueError('matched update schedule required')
    return dict(recordings=len(ids),updates=updates,schedule=batches['D'],Q_P_identical_inputs=True,
        same_final_labels=True,same_outer_weights=True,span_weights={'D':[1.],'Q':[0.,0.,1.],'P':[.1,.1,1.]})


def prepare(root,d_training,d_suite):
    root,d_training,d_suite=map(Path,(root,d_training,d_suite));read=lambda p:json.loads(p.read_text())
    if root.exists():raise FileExistsError('immutable matched-arm plan exists')
    cfg=read(d_training/'config-template.json')
    if cfg['arm']!='D' or cfg['step']!=0:raise ValueError('fixed D initialization required')
    populations={a:load_pilot(cfg['pilot'],cfg['manifest_sha256'],a)[0] for a in ('D','Q','P')}
    audit=audit_arms(populations,cfg['seed'],12);root.mkdir()
    repo=Path(__file__).resolve().parents[2];plans={}
    for arm in ('Q','P'):
        train=root/(arm+'-training');suite=root/(arm+'-suite');plans[arm]={}
        for out,template in ((train,d_training),(suite,d_suite)):
            out.mkdir()
            for folder in ('src','tools'):shutil.copytree(repo/folder,out/'source'/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            manifest={str(p.relative_to(out/'source')):file_hash(p) for p in (out/'source').rglob('*') if p.is_file()}
            for p,h in manifest.items():
                if file_hash(repo/p)!=h:raise ValueError('source changed during snapshot')
            atomic_json(out/'source-manifest.json',manifest)
            current=read(template/'config-template.json');plan=read(template/'prepared.json')
            if (template/'evidence').exists():shutil.copytree(template/'evidence',out/'evidence')
            current['pins']={p.replace(str(template),str(out)):h for p,h in current['pins'].items() if not p.startswith(str(template/'source')+'/')}
            current['pins'].update({str(out/'source'/p):h for p,h in manifest.items()})
            if out==train:current['arm']=arm
            else:current.update(training_root=str(train),adapter=str(train/'run/adapter'),adapter_id=None)
            argv=[a.replace(str(template),str(out)) for a in plan['create_argv']]
            argv[argv.index('--name')+1]='multimodal-jev-'+root.name+'-'+out.name.lower()
            atomic_json(out/'config-template.json',current)
            plan.update(root=str(out),create_argv=argv,config_template_sha256=file_hash(out/'config-template.json'),
                source_manifest_sha256=file_hash(out/'source-manifest.json'),status='prepared_not_dispatched',
                scope=f'{arm} first-update preparation from same D step0 adapter, fresh Adam; no training/efficacy claim.')
            if out==suite:plan['training_root']=str(train)
            atomic_json(out/'known-bind-pins.json',validate_bind_pins(argv,current['pins']));atomic_json(out/'prepared.json',plan)
            plans[arm]['training' if out==train else 'suite']=str(out)
    report=dict(root=str(root),plans=plans,audit=audit,initial_adapter=cfg['initial_adapter'],acceptance_root=cfg['acceptance_root'],
        seed=cfg['seed'],lr=cfg['lr'],global_requests=8,total_updates_per_arm=12,dispatch_allowed=False,
        scope='Q/P first update and independent suite prepared only. D12 remains priority; no formal held-out or multi-seed protocol admission.',
        files={str(d_training/'config-template.json'):file_hash(d_training/'config-template.json'),str(Path(__file__).resolve()):file_hash(__file__)})
    atomic_json(root/'prepared.json',report)
    for p in [root,*root.rglob('*')]:p.chmod(0o777 if p.is_dir() else 0o644)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','d-training','d-suite','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();result=prepare(a.root,a.d_training,a.d_suite);atomic_json(Path(a.output),result)

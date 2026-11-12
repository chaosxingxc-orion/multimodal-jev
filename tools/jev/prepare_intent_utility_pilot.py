#!/usr/bin/env python3
"""Freeze matched correctness/utility/uniform controls and exact resume probes."""
import argparse
import copy
from pathlib import Path
import shutil
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.container_pins import validate_bind_pins
from multimodal_jev.intent_objective import identity
from multimodal_jev.intent_actor_inputs import schedule
from multimodal_jev.slurp_intent_pilot import load_pilot
from queue_span_native_actor_npu import read
from queue_span_dp4_resume import check_pins
from prepare_intent_lr_control import freeze_source
from prepare_intent_continuation_step import prepare as continuation

OBJECTIVES=[{'name':'correctness-v1'},{'name':'smoothed-utility-v1'},
            {'name':'uniform-credit-v1','scale':8.}]


def audit(reference,candidate):
    if (reference['arm']!='D' or reference['step']!=0 or reference['lr']!=1e-4
            or reference['global_requests']!=8 or reference.get('previous_root')
            or reference.get('initial_optimizer_checkpoint')):
        raise ValueError('original D initialization and fresh Adam required')
    if candidate.get('objective') not in OBJECTIVES:
        raise ValueError('registered objective required')
    identity(candidate)
    strip=lambda c:{k:v for k,v in c.items() if k not in ('pins','objective')}
    if strip(reference)!=strip(candidate):raise ValueError('non-objective training factor changed')
    return dict(reference_objective=identity(reference),candidate_objective=identity(candidate))


def prepare(root,training_template,suite_template):
    root,training_template,suite_template=map(Path,(root,training_template,suite_template))
    if root.exists():raise FileExistsError('immutable utility pilot exists')
    repo=Path(__file__).resolve().parents[2];reference=read(training_template/'config-template.json')
    check_pins(reference['pins']);check_pins(read(suite_template/'config-template.json')['pins'])
    population,_=load_pilot(reference['pilot'],reference['manifest_sha256'],'D')
    root.mkdir();arms=[];audits=[]
    for objective in OBJECTIVES:
        label=objective['name'].removesuffix('-v1');arm_root=root/label;arm_root.mkdir()
        train,suite=arm_root/'training01',arm_root/'suite01'
        for out,template in ((train,training_template),(suite,suite_template)):
            out.mkdir();manifest=freeze_source(repo,template,out/'source','lr')
            atomic_json(out/'source-manifest.json',manifest)
            cfg=read(template/'config-template.json');plan=read(template/'prepared.json')
            if (template/'evidence').exists():shutil.copytree(template/'evidence',out/'evidence')
            cfg['pins']={p.replace(str(template),str(out)):h for p,h in cfg['pins'].items()
                if not p.startswith(str(template/'source')+'/')}
            cfg['pins'].update({str(out/'source'/p):h for p,h in manifest.items()})
            if out==train:
                cfg['objective']=copy.deepcopy(objective);audits.append(audit(reference,cfg))
            else:cfg.update(training_root=str(train),adapter=str(train/'run/adapter'),adapter_id=None)
            argv=[a.replace(str(template),str(out)) for a in plan['create_argv']]
            argv[argv.index('--name')+1]='multimodal-jev-'+root.name+'-'+label+'-'+out.name
            atomic_json(out/'config-template.json',cfg)
            plan.update(root=str(out),create_argv=argv,status='prepared_not_dispatched',
                config_template_sha256=file_hash(out/'config-template.json'),
                source_manifest_sha256=file_hash(out/'source-manifest.json'),
                scope='Two-step objective engineering admission, no sustained or held-out efficacy claim.')
            if out==suite:plan['training_root']=str(train)
            atomic_json(out/'known-bind-pins.json',validate_bind_pins(argv,cfg['pins']))
            atomic_json(out/'prepared.json',plan)
        train2,suite2=arm_root/'training02',arm_root/'suite02'
        continuation(train2,suite2,train,suite,train,suite,source_mode='template')
        # Continuation helper normally names by basename; nested arms require
        # globally distinct Docker names before freezing the completed plan.
        for out in (train2,suite2):
            plan=read(out/'prepared.json');argv=plan['create_argv']
            argv[argv.index('--name')+1]='multimodal-jev-'+root.name+'-'+label+'-'+out.name
            atomic_json(out/'prepared.json',plan)
        arms.append(dict(name=label,objective=objective,steps=[
            dict(training_root=str(t),suite_root=str(s)) for t,s in ((train,suite),(train2,suite2))]))
    protocol=dict(schema='intent-objective-engineering-pilot-v1',arms=arms,
        initial_adapter=reference['initial_adapter'],initial_gate=reference['acceptance_root'],
        reference_training=str(training_template),reference_suite=str(suite_template),
        reference_config_sha256=file_hash(training_template/'config-template.json'),
        pilot=reference['pilot'],manifest_sha256=reference['manifest_sha256'],
        schedule=[[r['id'] for r in schedule(population,reference['seed'],i,8)] for i in range(2)],
        seed=reference['seed'],lr=reference['lr'],global_requests=8,updates_per_arm=2,
        uniform_scale_rationale='Fixed8 approximates step0 all-learning output-logit norm sum ratio9.339/1.133. Not per-batch matching, not target-test selected; amplitude control only.',
        stage='Two-step engineering admission before separately registering full learning curves.',
        selection='Report every arm and both updates, no best-point selection.',
        reference_replay='Correctness arm shares current source with utility/control; compare first two updates with accepted original D states.',
        scope='Post-D12 prospective optimization hypothesis; no independent evaluation population yet. Fresh Adam and identical BF16 initialization; no RL/SFT comparison.')
    atomic_json(root/'protocol.json',protocol)
    report=dict(root=str(root),protocol_sha256=file_hash(root/'protocol.json'),arms=arms,audits=audits,
        status='prepared_not_dispatched',scope=protocol['scope'])
    atomic_json(root/'prepared.json',report)
    for p in [root,*root.rglob('*')]:p.chmod(0o777 if p.is_dir() else 0o644)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','training-template','suite-template'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(prepare(**vars(a)))

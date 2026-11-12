#!/usr/bin/env python3
"""Prepare a one-factor learning-rate control from the original RL start."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.container_pins import validate_bind_pins
from multimodal_jev.slurp_intent_pilot import load_pilot
from multimodal_jev.intent_actor_inputs import schedule
from queue_span_dp4_resume import check_pins


def audit_control(reference, control):
    if reference['arm']!='D' or reference['step']!=0 or reference['lr']!=1e-4:
        raise ValueError('fixed D first-update reference required')
    if control['lr']!=2e-5:raise ValueError('prospective five-fold lower learning rate required')
    # Relocated artifact pins are independently checked; all algorithm/config
    # fields apart from LR must be byte-for-byte equivalent JSON values.
    left={k:v for k,v in reference.items() if k not in ('lr','pins')}
    right={k:v for k,v in control.items() if k not in ('lr','pins')}
    if left!=right:raise ValueError('non-LR training factor changed')
    if control.get('previous_root') or control.get('initial_optimizer_checkpoint'):
        raise ValueError('fresh optimizer from original initialization required')
    return dict(only_training_parameter_changed='lr',reference_lr=reference['lr'],control_lr=control['lr'])


def prepare(root, reference_training, reference_suite, reference_protocol):
    root,reference_training,reference_suite,reference_protocol=map(Path,(root,reference_training,reference_suite,reference_protocol))
    read=lambda p:json.loads(p.read_text());repo=Path(__file__).resolve().parents[2]
    if root.exists():raise FileExistsError('immutable LR control already exists')
    reference=read(reference_training/'config-template.json');protocol=read(reference_protocol)
    candidate=copy.deepcopy(reference);candidate['lr']=2e-5;audit=audit_control(reference,candidate)
    items,_=load_pilot(reference['pilot'],reference['manifest_sha256'],'D')
    batches=[[r['id'] for r in schedule(items,reference['seed'],i,8)] for i in range(12)]
    if (protocol['schedule']!=batches or protocol['lr']!=reference['lr']
            or protocol['seed']!=reference['seed'] or protocol['total_updates']!=12):
        raise ValueError('exact D12 schedule reference required')
    check_pins(reference['pins']);root.mkdir()
    # Freeze the current implementation independently; compute-source differences
    # against the reference must be audited before dispatch.
    train=root/'training';suite=root/'suite'
    for out,template in ((train,reference_training),(suite,reference_suite)):
        out.mkdir()
        for folder in ('src','tools'):
            shutil.copytree(repo/folder,out/'source'/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        manifest={str(p.relative_to(out/'source')):file_hash(p) for p in (out/'source').rglob('*') if p.is_file()}
        for p,h in manifest.items():
            if file_hash(repo/p)!=h:raise ValueError('source changed during preparation')
        atomic_json(out/'source-manifest.json',manifest)
        cfg=read(template/'config-template.json');plan=read(template/'prepared.json')
        if (template/'evidence').exists():shutil.copytree(template/'evidence',out/'evidence')
        cfg['pins']={p.replace(str(template),str(out)):h for p,h in cfg['pins'].items() if not p.startswith(str(template/'source')+'/')}
        cfg['pins'].update({str(out/'source'/p):h for p,h in manifest.items()})
        if out==train:
            cfg['lr']=2e-5;audit_control(reference,cfg)
        else:cfg.update(training_root=str(train),adapter=str(train/'run/adapter'),adapter_id=None)
        argv=[a.replace(str(template),str(out)) for a in plan['create_argv']]
        argv[argv.index('--name')+1]='multimodal-jev-'+root.name+'-'+out.name
        atomic_json(out/'config-template.json',cfg)
        plan.update(root=str(out),create_argv=argv,status='prepared_not_dispatched',
            config_template_sha256=file_hash(out/'config-template.json'),source_manifest_sha256=file_hash(out/'source-manifest.json'),
            scope='Prospective lower-LR D first update and independent acceptance; no training dispatched.')
        if out==suite:plan['training_root']=str(train)
        atomic_json(out/'known-bind-pins.json',validate_bind_pins(argv,cfg['pins']));atomic_json(out/'prepared.json',plan)
    locked=dict(protocol,schema='intent-finite-d12-lr-control-v1',lr=2e-5,initial_updates=0,
        retrospective_updates=[],prospective_updates=list(range(1,13)),reference_protocol=str(reference_protocol),
        reference_protocol_sha256=file_hash(reference_protocol),
        rationale='First two reference updates showed concentrated credit and substantial post-update drift. Test one lower LR; no target/development feedback.',
        selection='Report all0-12 points; no best learning checkpoint selection or stopping for favorable learning reward.',
        scope='Prospective learning-only control, chosen after observing reference updates1/2. Not a blind hyperparameter study, held-out experiment or transfer result.')
    atomic_json(root/'protocol.json',locked)
    report=dict(root=str(root),training=str(train),suite=str(suite),audit=audit,
        protocol_sha256=file_hash(root/'protocol.json'),prepared_updates=1,total_planned_updates=12,
        continuation='Prepare steps2-12 with the same lower LR after first pair review; no mixed-LR resume.',
        execution_order='After existing D12 and same-LR Q/P pilot controls; no NPU dispatch in this preparation.',
        initial_adapter=reference['initial_adapter'],source_gate=reference['acceptance_root'],
        scope=locked['scope'])
    atomic_json(root/'prepared.json',report)
    for p in [root,*root.rglob('*')]:p.chmod(0o777 if p.is_dir() else 0o644)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','reference-training','reference-suite','reference-protocol','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();atomic_json(Path(a.output),prepare(a.root,a.reference_training,a.reference_suite,a.reference_protocol))

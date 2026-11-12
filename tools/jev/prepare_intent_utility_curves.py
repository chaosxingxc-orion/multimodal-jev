#!/usr/bin/env python3
"""Prospectively prepare all three objective curves through update12.

Extends each existing two-update arm without changing its source or optimizer.
No launch, no checkpoint selection, no independent efficacy claim.
"""
import argparse
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_actor_inputs import schedule
from multimodal_jev.intent_resume import validate_lineage
from multimodal_jev.slurp_intent_pilot import load_pilot
from queue_span_native_actor_npu import read
from queue_span_dp4_resume import check_pins
from prepare_intent_utility_pilot import OBJECTIVES
from prepare_intent_continuation_step import prepare as prepare_step


def validate_protocol(protocol, pilot):
    if (protocol['schema'] != 'intent-objective-curves-v1'
            or pilot['schema'] != 'intent-objective-engineering-pilot-v1'
            or pilot['updates_per_arm'] != 2 or protocol['initial_updates'] != 2
            or protocol['total_updates'] != 12 or protocol['source_uses_per_arm'] != 96
            or protocol['checkpoints'] != list(range(13))
            or [a['objective'] for a in protocol['arms']] != OBJECTIVES
            or [a['objective'] for a in pilot['arms']] != OBJECTIVES):
        raise ValueError('complete fixed three-arm12-update protocol required')
    for key in ('initial_adapter','initial_gate','pilot','manifest_sha256','seed','lr','global_requests'):
        if protocol[key] != pilot[key]:raise ValueError('pilot factor changed: '+key)
    if (protocol['global_requests'] != 8 or len(protocol['schedule']) != 12
            or protocol['schedule'][:2] != pilot['schedule']):
        raise ValueError('unchanged full source schedule required')
    for batch in protocol['schedule']:
        if len(batch) != 8 or len(set(batch)) != 8:raise ValueError('eight distinct sources per update required')
    seen=set()
    for current,initial in zip(protocol['arms'],pilot['arms'],strict=True):
        if current['name'] != initial['name'] or current['steps'][:2] != initial['steps'] or len(current['steps']) != 12:
            raise ValueError('preserve both initial updates and every continuation')
        for step in current['steps']:
            for key in ('training_root','suite_root'):
                if step[key] in seen:raise ValueError('distinct outputs required')
                seen.add(step[key])


def prepare(pilot_root,output):
    pilot_root,output=map(Path,(pilot_root,output));pilot=read(pilot_root/'protocol.json')
    if output.exists():raise FileExistsError('immutable curve protocol already exists')
    if (pilot['updates_per_arm'] != 2 or [a['objective'] for a in pilot['arms']] != OBJECTIVES):
        raise ValueError('fixed original objective pilot required')
    items,_=load_pilot(pilot['pilot'],pilot['manifest_sha256'],'D')
    batches=[[r['id'] for r in schedule(items,pilot['seed'],step,8)] for step in range(12)]
    if batches[:2] != pilot['schedule']:raise ValueError('pilot schedule differs')
    # Register while all objective arms are still unlaunched, before outcomes.
    for arm in pilot['arms']:
        for step in arm['steps']:
            for kind in ('training_root','suite_root'):
                path=Path(step[kind])
                if any((path/n).exists() for n in ('config.json','launch.json','create-attempt.json','run')):
                    raise ValueError('prospective registration requires unlaunched pilot')
                check_pins(read(path/'config-template.json')['pins'])
                check_pins({str(path/'source'/p):h for p,h in read(path/'source-manifest.json').items()})
    output.mkdir();arms=[];pins={str(pilot_root/'protocol.json'):file_hash(pilot_root/'protocol.json')};names=set()
    for arm in pilot['arms']:
        steps=list(arm['steps']);template=steps[0]
        for index in range(2,12):
            train=output/f'{arm["name"]}-training{index+1:02d}'
            suite=output/f'{arm["name"]}-suite{index+1:02d}'
            prepare_step(train,suite,steps[-1]['training_root'],steps[-1]['suite_root'],
                         template['training_root'],template['suite_root'],source_mode='template')
            steps.append(dict(training_root=str(train),suite_root=str(suite)))
        previous=None
        for index,step in enumerate(steps):
            for kind in ('training_root','suite_root'):
                root=Path(step[kind]);plan=read(root/'prepared.json');cfg=read(root/'config-template.json')
                name=plan['create_argv'][plan['create_argv'].index('--name')+1]
                if name in names:raise ValueError('duplicate container name')
                names.add(name)
                if file_hash(root/'source-manifest.json') != file_hash(Path(template[kind])/'source-manifest.json'):
                    raise ValueError('curve changes frozen computation source')
                for filename in ('prepared.json','config-template.json','source-manifest.json'):
                    pins[str(root/filename)]=file_hash(root/filename)
                if kind=='training_root':
                    if cfg['step'] != index or cfg['objective'] != arm['objective']:
                        raise ValueError('curve objective or cursor changed')
                    if previous is not None:validate_lineage(previous,cfg)
                    previous=cfg
        arms.append(dict(name=arm['name'],objective=arm['objective'],steps=steps))
    protocol={k:pilot[k] for k in ('initial_adapter','initial_gate','pilot','manifest_sha256','seed','lr','global_requests')}
    protocol.update(schema='intent-objective-curves-v1',pilot_protocol=str(pilot_root/'protocol.json'),
        pilot_protocol_sha256=file_hash(pilot_root/'protocol.json'),arms=arms,initial_updates=2,total_updates=12,
        source_uses_per_arm=96,checkpoints=list(range(13)),schedule=batches,
        selection='All three arms and every checkpoint, no best-learning-point selection or outcome-based early stopping.',
        admission='Each pilot must pass two actual updates, independent gates and restoration; correctness replay must match original D2 before extension.',
        scope='Prospective learning-population optimization study. Independent development admission pending; no generalization claim.')
    validate_protocol(protocol,pilot);atomic_json(output/'protocol.json',protocol)
    pins[str(output/'protocol.json')]=file_hash(output/'protocol.json')
    result=dict(root=str(output),status='prepared_not_dispatched',new_updates=30,new_training_suite_pairs=30,
        source_manifest_preserved=True,pins=pins,scope=protocol['scope'])
    atomic_json(output/'prepared.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.pilot_root,a.output)

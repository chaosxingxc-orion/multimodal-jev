#!/usr/bin/env python3
"""Compile semantic actor menu design, without rendering or dispatching training."""
import argparse
from collections import Counter
from pathlib import Path
import json

from consolidate_intent_presence_references import catalog, read, rows, sha, require
from audit_slurp_menu_design import keyed


def compile_panels(intent_ids, slot_ids, width=4, seed=317):
    require(type(width) is int and width > 0, 'positive width required')
    orders = {}
    for family, values in [('I_present', intent_ids), ('S_present', slot_ids)]:
        require(values and len(set(values)) == len(values), 'nonempty unique definitions required')
        orders[family] = sorted(values, key=lambda k: (keyed('semantic-actor-menu-v1', seed, family, k), k))
    count = max((len(v)+width-1)//width for v in orders.values())
    panels = []
    for index in range(count):
        selected = {f: v[index*width:(index+1)*width] for f,v in orders.items()}
        q = [dict(operator=f, definition_id=selected[f][j]) for j in range(width)
             for f in ('I_present','S_present') if j < len(selected[f])]
        panels.append(dict(panel_id=f'semantic-{index:02d}', process_questions=q,
            configured_counts={f:len(v) for f,v in selected.items()},
            final='complete_native_60_class_choice',
            arms=dict(Direct=[], Q_input=q, Q_credit=q)))
    return panels


def build(intent_catalog, slot_package, release, output):
    require(not output.exists(), 'output exists')
    pins={};cards,_=catalog(intent_catalog,pins)
    package=read(slot_package);pins[str(slot_package)]=sha(slot_package)
    for k in ('definitions','policy','boundaries'):
        pins[package[k]]=sha(package[k])
    slots=read(package['definitions'])['types']
    slot_ids=[x['entity_type'] for x in slots]
    require(len(cards)==65 and len(slot_ids)==55, 'current full ontology required')
    manifest=read(release/'MANIFEST.json');pins[str(release/'MANIFEST.json')]=sha(release/'MANIFEST.json')
    for n in ('learning.jsonl','selection.jsonl','tasks.json'):
        require(sha(release/n)==manifest['files'][n], 'changed release '+n)
        pins[str(release/n)]=sha(release/n)
    learning,selection=rows(release/'learning.jsonl'),rows(release/'selection.jsonl')
    native=read(release/'tasks.json')['intent']['candidate_pairs']
    require(len(native)==60 and len(learning)==40513 and len(selection)==9226,'population mismatch')
    require(set(r['source_group'] for r in learning).isdisjoint(r['source_group'] for r in selection),'source leakage')
    panels=compile_panels(list(cards),slot_ids)
    require(len(panels)==17,'panel count mismatch')
    for f,ids in [('I_present',cards),('S_present',slot_ids)]:
        c=Counter(q['definition_id'] for p in panels for q in p['process_questions'] if q['operator']==f)
        require(c==Counter({k:1 for k in ids}),'coverage mismatch')
    output.mkdir(parents=True)
    (output/'panels.json').write_text(json.dumps(panels,indent=2)+'\n')
    # Separate native classification descriptors from semantic card IDs.
    (output/'native-menu.json').write_text(json.dumps([
        dict(native_id=k,scenario=v[0],action=v[1],descriptor=v[0]+' / '+v[1])
        for k,v in sorted(native.items())],indent=2)+'\n')
    report=dict(schema='jev-semantic-actor-menu-design-v1',pins=pins,
        implementation_sha256=sha(__file__),files={n:sha(output/n) for n in ('panels.json','native-menu.json')},
        seed=317,panel_width_per_family=4,panels=len(panels),
        coverage=dict(native_Final=60,I_present=65,S_present=55,occurrences_per_predicate_per_sweep=1),
        learning=dict(recordings=len(learning),source_groups=len({r['source_group'] for r in learning}),
            requests_per_complete_sweep=len(learning)*len(panels)),
        selection=dict(recordings=len(selection),requests_per_arm_per_checkpoint=len(selection)*len(panels),
            three_arms_canonical_requests=len(selection)*len(panels)*3),
        design=dict(selection='All fixed panels for every audio, independent of native/semantic labels and reference text.',
            knowledge='Complete native60 descriptor menu plus selected panel semantic definitions and all required shared/incident boundary rules, identical in Direct/Q_input/Q_credit for that panel.',
            final='One model Final per call, reported separately per panel and equal-panel mean; no vote/union/best-panel choice.',
            semantics='Native categories are classification descriptors, not semantic predicate definitions. All-false semantic outputs do not imply OOD.',
            normalization='Fixed complete-sweep denominators; incomplete last panels are not padded with duplicate semantic questions.',
            context='Four per family is inherited candidate width, not accepted context. Changing width requires recompiling schedule,objective and evaluation counts.'),
        status='task_schedule_candidate_not_renderer_or_runtime_release',
        training_dispatch_allowed=False,evaluation_dispatch_allowed=False,semantic_admission=0)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('intent-catalog','slot-package','release','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();r=build(a.intent_catalog,a.slot_package,a.release,a.output)
    print(json.dumps({k:v for k,v in r.items() if k!='pins'},indent=2))

#!/usr/bin/env python3
"""Materialize label-independent task menu candidates; no actor or trainer changes."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def keyed(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def panel(orders, group, epoch, seed, stride):
    """Only opaque group ID, fixed definitions, epoch and seed enter selection."""
    result = {}
    for family, order in orders.items():
        start = (int(keyed('menu-source-v1', seed, family, group), 16)+stride*epoch) % len(order)
        result[family] = [order[(start+i) % len(order)] for i in range(4)]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--slot-report', type=Path, required=True)
    p.add_argument('--slot-report-sha256', required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if sha(a.release/'MANIFEST.json') != a.manifest_sha256 or sha(a.slot_report) != a.slot_report_sha256:
        raise ValueError('pinned population identity required')
    manifest, slot_report, cfg = read(a.release/'MANIFEST.json'), read(a.slot_report), read(a.config)
    if cfg['training_dispatch_allowed'] is not False or cfg['schema'] != 'slurp-menu-design-candidate-v2':
        raise ValueError('design-only config required')
    pins = {str(a.config): sha(a.config), str(a.release/'MANIFEST.json'): a.manifest_sha256,
            str(a.release/'learning.jsonl'): manifest['files']['learning.jsonl'],
            str(a.slot_report): a.slot_report_sha256,
            str(a.slot_report.parent/'records.jsonl'): slot_report['records_sha256']}
    for path, digest in slot_report['pins'].items():
        pins[path] = digest
    for family in cfg['families'].values():
        for k in ('definitions','policy'):
            if k in family:
                pins[family[k]] = sha(family[k])
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError(f'changed dependency {path}')
    intent = read(cfg['families']['intent']['definitions'])['cards']
    slot = read(cfg['families']['slot']['definitions'])['types']
    ids = dict(intent=[c['semantic_id'] for c in intent], slot=[c['entity_type'] for c in slot])
    if len(set(ids['intent'])) != 60 or len(set(ids['slot'])) != 55:
        raise ValueError('complete definition families required')
    seed = cfg['seed']
    orders = {f:sorted(values, key=lambda x:keyed('menu-order-v1',seed,f,x)) for f,values in ids.items()}
    stride = cfg['rotation_stride']
    if type(stride) is not int or stride < 1 or any(math.gcd(stride,len(order)) != 1 for order in orders.values()):
        raise ValueError('rotation must eventually expose each definition to every source')
    learning = [json.loads(line) for line in (a.release/'learning.jsonl').open()]
    slots = {r['id']:r for r in map(json.loads,(a.slot_report.parent/'records.jsonl').open())}
    if len(learning) != 40513 or set(slots) != {r['id'] for r in learning}:
        raise ValueError('full population required')
    groups = sorted({r['source_group'] for r in learning})
    # Compile all panels before looking at any label or native slot annotation.
    compiled = {(g,e):panel(orders,g,e,seed,stride) for g in groups for e in range(cfg['audit_epochs'])}
    exposures, positives = defaultdict(Counter), defaultdict(Counter)
    source_exposures, source_positives = defaultdict(lambda:defaultdict(set)), defaultdict(lambda:defaultdict(set))
    panel_positive_hist = defaultdict(Counter)
    class_included = defaultdict(Counter)
    a.output.mkdir(parents=True)
    with (a.output/'panels.jsonl').open('w') as out:
        for (group,epoch), selected in compiled.items():
            out.write(json.dumps(dict(source_group=group,epoch=epoch,q8=selected,
                                      q2={f:v[:1] for f,v in selected.items()}),sort_keys=True)+'\n')
    for row in learning:
        if row['role'] != 'learning' or row['known_pending_source'] or row['official_split'] != 'train':
            raise ValueError('learning role required')
        if slots[row['id']]['source_group'] != row['source_group']:
            raise ValueError('slot source mismatch')
        for epoch in range(cfg['audit_epochs']):
            selected = compiled[(row['source_group'],epoch)]
            native = dict(intent={row['intent']},slot=set(slots[row['id']]['annotated_types']))
            for arm, counts in cfg['arms'].items():
                total_positive = 0
                for family in orders:
                    chosen = selected[family][:counts[family+'_questions']]
                    key = arm+':'+family
                    for definition in chosen:
                        exposures[key][definition] += 1
                        source_exposures[key][definition].add(row['source_group'])
                        if definition in native[family]:
                            positives[key][definition] += 1
                            source_positives[key][definition].add(row['source_group'])
                            total_positive += 1
                    if family == 'intent' and row['intent'] in chosen:
                        class_included[arm][row['intent']] += 1
                panel_positive_hist[arm][total_positive] += 1
    reports = {}
    for arm in cfg['arms']:
        families = {}
        for family, values in ids.items():
            k = arm+':'+family
            families[family] = [dict(semantic_id=t,recording_epoch_exposures=exposures[k][t],
                native_positive_recording_epochs=positives[k][t],
                unique_source_groups=len(source_exposures[k][t]),
                native_positive_source_groups=len(source_positives[k][t])) for t in sorted(values)]
        reports[arm] = dict(families=families,native_positive_Q_count_histogram=dict(sorted(panel_positive_hist[arm].items())),
            intent_classes_with_no_positive_Q=[t for t in ids['intent'] if class_included[arm][t]==0])
    report = dict(schema='slurp-menu-design-audit-v1',pins=pins,implementation_sha256=sha(__file__),
        recordings=len(learning),source_groups=len(groups),audit_epochs=cfg['audit_epochs'],orders=orders,
        panels_sha256=sha(a.output/'panels.jsonl'),arms=reports,
        scope='Native annotation projection only, not semantic positive/negative supervision or token/runtime acceptance. Same-source recordings receive identical menus. No training dispatch.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for arm,r in reports.items():
        print(json.dumps(dict(arm=arm,native_positive_Q_histogram=r['native_positive_Q_count_histogram'],
            zero_positive_intent_classes=r['intent_classes_with_no_positive_Q'],
            totals={f:dict(exposures=sum(x['recording_epoch_exposures'] for x in vals),
                           native_positive=sum(x['native_positive_recording_epochs'] for x in vals)) for f,vals in r['families'].items()})))


if __name__ == '__main__':
    main()

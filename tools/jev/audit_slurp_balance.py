#!/usr/bin/env python3
"""Audit full learning-population balance; never emit training labels or launch jobs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def class_weights(counts, mode):
    if not counts or any(type(n) is not int or n <= 0 for n in counts.values()):
        raise ValueError('positive integer counts required')
    n, c = sum(counts.values()), len(counts)
    if mode == 'natural':
        raw = {k: 1.0 for k in counts}
    elif mode == 'inverse_frequency':
        raw = {k: n/(c*v) for k, v in counts.items()}
    elif mode == 'sqrt_cap4':
        raw = {k: min(4.0, math.sqrt(n/(c*v))) for k, v in counts.items()}
    else:
        raise ValueError('unknown balance mode')
    z = math.fsum(counts[k]*raw[k] for k in counts)/n
    return {k: v/z for k, v in raw.items()}


def effective_size(weights):
    return math.fsum(weights)**2/math.fsum(w*w for w in weights)


def audit(rows, orders, compiled, epochs):
    counts = Counter(r['intent'] for r in rows)
    if set(counts) != set(orders['intent']):
        raise ValueError('complete intent ontology required')
    groups = defaultdict(list)
    for row in rows:
        groups[row['source_group']].append(row)
    n = len(rows)
    profiles = {}
    for mode in ('natural', 'sqrt_cap4', 'inverse_frequency'):
        weights = class_weights(counts, mode)
        values = [weights[r['intent']] for r in rows]
        total = math.fsum(values)
        group_mass = {g: math.fsum(weights[r['intent']] for r in rs) for g, rs in groups.items()}
        per_class = {}
        for intent, count in sorted(counts.items()):
            supports = [g for g, rs in groups.items() if any(r['intent'] == intent for r in rs)]
            per_class[intent] = dict(recordings=count, source_groups=len(supports),
                weight=weights[intent], objective_mass=count*weights[intent]/total,
                complete_I_native_positive_fraction=count*weights[intent]/total)
        menus = {}
        for width in (1, 4):
            positive_mass = 0.0
            exposure_mass = 0.0
            class_positive = Counter()
            zero_positive = 0.0
            for row in rows:
                for epoch in range(epochs):
                    chosen = compiled[(row['source_group'], epoch)]['intent'][:width]
                    exposure_mass += weights[row['intent']]*len(chosen)
                    if row['intent'] in chosen:
                        positive_mass += weights[row['intent']]
                        class_positive[row['intent']] += weights[row['intent']]
                    else:
                        zero_positive += weights[row['intent']]
            menus[str(width)+'I'] = dict(
                weighted_native_positive_fraction=positive_mass/exposure_mass,
                weighted_no_native_positive_fraction=zero_positive/(total*epochs),
                classes_with_zero_native_positive=sorted(set(counts)-set(class_positive)))
        profiles[mode] = dict(
            normalized_weight_min=min(values), normalized_weight_max=max(values),
            mean_weight=total/n, recording_weight_ESS=effective_size(values),
            source_mass_ESS=effective_size(list(group_mass.values())),
            largest_source_mass=max(group_mass.values())/total,
            complete_I_native_positive_fraction=total/(total*len(counts)),
            complete_I_always_false_native_accuracy=1-1/len(counts),
            per_class=per_class, rotating_menus=menus)
    return dict(recordings=n, source_groups=len(groups), classes=len(counts), profiles=profiles,
        cautions=[
            'Native intent projection only; no semantic Q positive/negative admission.',
            'Weights describe a changed objective, not additional independent audio or sources.',
            'ESS measures weight concentration, not independent sample count or confidence intervals.',
            'Outer class weights cannot change aggregate complete-I positive prevalence of 1/C.',
            'Weighting cannot repair zero positive exposure in a fixed question schedule.',
            'Cap4 applies before normalization; normalized weights can exceed4.',
            'No selection/test examples or statistics used; no resampling or training performed.'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--menu-report', type=Path, required=True)
    p.add_argument('--menu-report-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if sha(a.release/'MANIFEST.json') != a.manifest_sha256 or sha(a.menu_report) != a.menu_report_sha256:
        raise ValueError('pinned sources required')
    manifest = json.loads((a.release/'MANIFEST.json').read_text())
    menu = json.loads(a.menu_report.read_text())
    learning = a.release/'learning.jsonl'
    panels = a.menu_report.parent/'panels.jsonl'
    pins = {str(a.release/'MANIFEST.json'):a.manifest_sha256,
            str(learning):manifest['files']['learning.jsonl'],
            str(a.menu_report):a.menu_report_sha256, str(panels):menu['panels_sha256']}
    pins.update(menu['pins'])
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError(f'changed dependency: {path}')
    rows = list(map(json.loads, learning.read_text().splitlines()))
    if len(rows) != 40513 or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('unique full population required')
    if any(r['role'] != 'learning' or r['official_split'] != 'train' or r['known_pending_source'] for r in rows):
        raise ValueError('admitted learning population required')
    schedule = list(map(json.loads, panels.read_text().splitlines()))
    compiled = {(r['source_group'], r['epoch']):r['q8'] for r in schedule}
    expected = {(r['source_group'], e) for r in rows for e in range(menu['audit_epochs'])}
    if len(compiled) != len(schedule) or set(compiled) != expected:
        raise ValueError('exact source/epoch coverage required')
    result = audit(rows, menu['orders'], compiled, menu['audit_epochs'])
    if {k:v['recordings'] for k,v in result['profiles']['natural']['per_class'].items()} != manifest['counts']['learning']['classes']:
        raise ValueError('manifest class counts disagree')
    result.update(schema='slurp-full-population-balance-audit-v1', training_dispatch_allowed=False,
                  pins=pins, implementation_sha256=sha(__file__))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:{f:v for f,v in r.items() if f != 'per_class'} for k,r in result['profiles'].items()},indent=2))


if __name__ == '__main__':
    main()

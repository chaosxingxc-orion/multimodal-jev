#!/usr/bin/env python3
"""Analyze bounded polarity CE coefficients; native projections are not new labels."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def polarity_weights(positive_mass, negative_mass, cap):
    if (any(not math.isfinite(x) or x < 0 for x in (positive_mass,negative_mass))
            or not math.isfinite(cap) or cap < 1):
        raise ValueError('finite nonnegative support and cap>=1 required')
    total=positive_mass+negative_mass
    return tuple(min(cap,total/(2*x)) if x else 0.0 for x in (positive_mass,negative_mass))


def profile(positive_mass,negative_mass,total_population_mass,max_positive_weight,max_negative_weight,cap):
    bp,bn=polarity_weights(positive_mass,negative_mass,cap)
    p,n=positive_mass*bp,negative_mass*bn
    return dict(positive_coefficient=bp,negative_coefficient=bn,
        effective_positive_share=p/(p+n) if p+n else None,
        retained_definition_mass=(p+n)/total_population_mass,
        valid_coverage=(positive_mass+negative_mass)/total_population_mass,
        max_outer_times_polarity=max(max_positive_weight*bp,max_negative_weight*bn),
        missing_polarity=positive_mass==0 or negative_mass==0)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--balance-report',type=Path,required=True)
    p.add_argument('--balance-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if sha(a.balance_report)!=a.balance_sha256:raise ValueError('balance report identity mismatch')
    balance=json.loads(a.balance_report.read_text())
    for path,h in balance['pins'].items():
        if sha(path)!=h:raise ValueError('changed dependency '+path)
    learning=next(Path(k) for k in balance['pins'] if k.endswith('/official-clean-intent-v1/learning.jsonl'))
    slots=next(Path(k) for k in balance['pins'] if k.endswith('/slot-support-audit-v1/records.jsonl'))
    slot_report=next(Path(k) for k in balance['pins'] if k.endswith('/slot-support-audit-v1/report.json'))
    rows=list(map(json.loads,learning.read_text().splitlines()))
    native_slots={r['id']:set(r['annotated_types']) for r in map(json.loads,slots.read_text().splitlines())}
    if set(native_slots)!={r['id'] for r in rows}:raise ValueError('slot population mismatch')
    class_counts=Counter(r['intent'] for r in rows);N=len(rows);C=len(class_counts)
    raw={k:min(4,math.sqrt(N/(C*n))) for k,n in class_counts.items()}
    z=math.fsum(raw[k]*v for k,v in class_counts.items())/N
    weights={k:v/z for k,v in raw.items()}
    for k,w in weights.items():
        if not math.isclose(w,balance['profiles']['sqrt_cap4']['per_class'][k]['weight'],rel_tol=1e-12):raise ValueError('outer weight mismatch')
    total=math.fsum(weights[r['intent']] for r in rows)
    families={'I':sorted(class_counts),'S':sorted(r['entity_type'] for r in json.loads(slot_report.read_text())['per_type'])}
    definitions={}
    for family,ids in families.items():
        data=[]
        for key in ids:
            positive=[];negative=[];pos_groups=set()
            for r in rows:
                match=r['intent']==key if family=='I' else key in native_slots[r['id']]
                (positive if match else negative).append(weights[r['intent']])
                if match:pos_groups.add(r['source_group'])
            pm,nm=math.fsum(positive),math.fsum(negative)
            # Exact inverse-frequency is a concentration diagnostic; cap>N emulates no cap here.
            profiles={str(cap):profile(pm,nm,total,max(positive,default=0),max(negative,default=0),cap) for cap in (8,32,N)}
            data.append(dict(definition_id=key,native_positive_recordings=len(positive),native_absent_recordings=len(negative),
                native_positive_source_groups=len(pos_groups),positive_mass=pm,negative_mass=nm,profiles=profiles))
        definitions[family]=data
    summary={}
    for family,data in definitions.items():
        summary[family]={}
        for cap in (8,32,N):
            vals=[r['profiles'][str(cap)] for r in data]
            summary[family][str(cap)]=dict(
                mean_retained_mass=math.fsum(v['retained_definition_mass'] for v in vals)/len(vals),
                min_positive_share=min(v['effective_positive_share'] for v in vals),
                max_positive_share=max(v['effective_positive_share'] for v in vals),
                largest_outer_times_polarity=max(v['max_outer_times_polarity'] for v in vals),
                positive_saturated_definitions=sum(v['positive_coefficient']==cap for v in vals))
    out=dict(schema='stage1-polarity-budget-audit-v1',balance_report=str(a.balance_report),balance_sha256=a.balance_sha256,
        implementation_sha256=sha(__file__),recordings=N,source_groups=len({r['source_group'] for r in rows}),
        caps=[8,32,N],uncapped_equivalent= N,definitions=definitions,summary=summary,
        training_dispatch_allowed=False,
        scope='Native projection stress test only; S absent is NOT a semantic negative. No coefficients or gold admitted for training. Actual semantic counts must replace these projections before freeze.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

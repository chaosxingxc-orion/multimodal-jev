#!/usr/bin/env python3
"""Learning-only native annotation information audit; no model or semantic labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_rows(p):
    return [json.loads(line) for line in Path(p).read_text().splitlines()]


def entropy(counts):
    total=sum(counts)
    return -sum(n/total*math.log2(n/total) for n in counts if n) if total else 0.


def fold(group):
    return int(hashlib.sha256(('question-information-v1:317:'+group).encode()).hexdigest(),16)%5


def signature(row, intents, slots):
    return tuple(row['intent']==i for i in intents)+tuple(s in row['types'] for s in slots)


def majority(counts):
    return min(counts, key=lambda label: (-counts[label],label))


def analyze(rows, intents=(), slots=()):
    if not rows or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('nonempty unique recording population required')
    if len(set(intents))!=len(intents) or len(set(slots))!=len(slots):
        raise ValueError('duplicate questions')
    cells=defaultdict(Counter); by_fold=defaultdict(lambda:defaultdict(Counter))
    classes=Counter(); fold_classes=defaultdict(Counter)
    for row in rows:
        key=signature(row,intents,slots); y=row['intent']; f=fold(row['source_group'])
        cells[key][y]+=1; by_fold[f][key][y]+=1
        classes[y]+=1; fold_classes[f][y]+=1
    n=len(rows)
    correct=unknown=0; per_class=Counter(); group_counts=defaultdict(lambda:[0,0])
    for row in rows:
        f=fold(row['source_group']); key=signature(row,intents,slots)
        train_counts=cells[key]-by_fold[f][key]
        if not train_counts:
            unknown+=1; train_counts=classes-fold_classes[f]
        if not train_counts:
            raise ValueError('cross-fit fold has no training sources')
        ok=majority(train_counts)==row['intent']
        correct+=ok; per_class[row['intent']]+=ok
        group_counts[row['source_group']][0]+=ok; group_counts[row['source_group']][1]+=1
    conditional=sum(sum(c.values())/n*entropy(c.values()) for c in cells.values())
    pure=sum(sum(c.values()) for c in cells.values() if len(c)==1)
    return dict(recordings=n, questions=len(intents)+len(slots), observed_signatures=len(cells),
        intent_entropy_bits=entropy(classes.values()), conditional_entropy_bits=conditional,
        empirical_mutual_information_bits=entropy(classes.values())-conditional,
        empirical_signature_majority_upper_bound=sum(max(c.values()) for c in cells.values())/n,
        pure_signature_recordings=pure,
        mixed_signature_recordings=n-pure,
        all_false_recordings=sum(cells.get((False,)*(len(intents)+len(slots)),{}).values()),
        source_group_5fold_lookup=dict(correct=correct, accuracy=correct/n,
            macro_recall=sum(per_class[y]/c for y,c in classes.items())/len(classes),
            source_group_equal_accuracy=sum(c/t for c,t in group_counts.values())/len(group_counts),
            unseen_signature_recordings=unknown,
            per_class={y:dict(support=c,correct=per_class[y]) for y,c in sorted(classes.items())}),
        cells=[dict(signature=list(key),intent_counts=dict(sorted(c.items()))) for key,c in sorted(cells.items())])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--support-report',type=Path,required=True)
    p.add_argument('--panels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    support=json.loads(a.support_report.read_text()); panels=json.loads(a.panels.read_text())
    pins=dict(support['pins'])
    pins.update({str(a.support_report):sha(a.support_report),str(a.panels):sha(a.panels),
                 str(a.support_report.parent/'records.jsonl'):support['records_sha256']})
    menu_path=Path(panels['menu_report'])
    pins[str(menu_path)]=panels['menu_report_sha256']
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('changed dependency: '+path)
    learning=read_rows(next(p for p in pins if Path(p).name=='learning.jsonl'))
    slots={r['id']:r for r in read_rows(a.support_report.parent/'records.jsonl')}
    if len(slots)!=len(learning) or set(slots)!={r['id'] for r in learning}:
        raise ValueError('slot population differs')
    rows=[]
    for r in learning:
        s=slots[r['id']]
        if r['role']!='learning' or r['official_split']!='train' or s['source_group']!=r['source_group']:
            raise ValueError('population role/group mismatch')
        rows.append(dict(id=r['id'],source_group=r['source_group'],intent=r['intent'],types=frozenset(s['annotated_types'])))
    all_i=[i for p in panels['panels'] for i in p['families']['intent']['definitions']]
    all_s=[s for p in panels['panels'] for s in p['families']['slot']['definitions']]
    if len(set(all_i))!=60 or len(all_i)!=60 or len(set(all_s))!=55 or len(all_s)!=55:
        raise ValueError('complete unique ontology required')
    if {r['intent'] for r in rows}!=set(all_i) or not set.union(*(set(r['types']) for r in rows))<=set(all_s):
        raise ValueError('unknown labels')
    results={'no_questions':analyze(rows),'all55_native_slots':analyze(rows,slots=all_s),
             'all60_native_intent_memberships':analyze(rows,intents=all_i)}
    for panel in panels['panels']:
        i=panel['families']['intent']['definitions']; s=panel['families']['slot']['definitions']
        for name,ii,ss in [('I',i,[]),('S',[],s),('IS',i,s)]:
            results[f"panel{panel['panel_id']:02d}_{name}"]=analyze(rows,ii,ss)
    a.output.mkdir(parents=True)
    cells={key:value.pop('cells') for key,value in results.items()}
    (a.output/'signature-counts.json').write_text(json.dumps(cells,indent=2)+'\n')
    summary=dict(schema='slurp-native-question-information-v1',pins=pins,implementation_sha256=sha(__file__),
        recordings=len(rows),source_groups=len({r['source_group'] for r in rows}),
        folds={str(f):dict(recordings=sum(fold(r['source_group'])==f for r in rows),
            groups=len({r['source_group'] for r in rows if fold(r['source_group'])==f})) for f in range(5)},
        results=results, panel_means={name:{metric:sum(results[f'panel{p:02d}_{name}'][metric] for p in range(15))/15
            for metric in ('empirical_signature_majority_upper_bound','conditional_entropy_bits')} for name in ('I','S','IS')},
        restrictions=['Native annotation signatures are descriptive only; omitted slots are not semantic negatives.',
            'In-sample signature-majority is an empirical upper bound only for deterministic mappings of this signature on this population, not for audio models or new data.',
            'Five folds partition source groups within learning; no heldout selection/test data or neural training.',
            'Lookup uses other folds recording counts; ties lexical; unseen signatures use other-fold global majority.',
            'I_class projections are Final label re-encodings, not independent semantic process truth.',
            'No questions selected using per-query labels; existing panels unchanged; full55 is structural diagnostic not single-call acceptance.',
            'Annotation correlation cannot establish causal Q use or RL benefit.'],
        training_dispatch_allowed=False,files={'signature-counts.json':sha(a.output/'signature-counts.json')})
    (a.output/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(recordings=len(rows),panel_means=summary['panel_means'],
        full_slots={k:v for k,v in results['all55_native_slots'].items() if k!='source_group_5fold_lookup'},
        full_slots_crossfit=results['all55_native_slots']['source_group_5fold_lookup']['accuracy']),indent=2))


if __name__=='__main__':main()

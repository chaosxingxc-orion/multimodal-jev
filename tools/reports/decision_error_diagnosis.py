"""Deterministic complete-population decision diagnosis and TRAIN-only support.

No model calls, threshold tuning, sample selection for training, or queue changes.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path

from multimodal_jev.context_event_metrics import ContextEventMetrics
from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnose(records, labels, candidates):
    truth = {r['id']: r for r in labels}
    if len(truth) != len(labels) or len(records) != len(labels) or {r['id'] for r in records} != set(truth):
        raise ValueError('complete unique prediction/label population required')
    scorer = ContextEventMetrics(candidates)
    outcomes = Counter(); confusion = Counter(); details = []; fp_by_class = Counter()
    for r in records:
        target = truth[r['id']]; gold = candidates.index(target['intent']); p = r['prob']
        if len(p) != len(candidates) or any(not math.isfinite(v) or not 0 < v < 1 for v in p):
            raise ValueError('finite interior candidate probabilities required')
        scorer.add(r['id'], target['intent'], p)
        chosen = [i for i, v in enumerate(p) if v >= .5]
        top = max(range(len(p)), key=p.__getitem__)
        outcome = ('exact' if chosen == [gold] else 'empty' if not chosen else
                   'gold_plus_extras' if gold in chosen else 'wrong_nonempty')
        negative = [i for i in range(len(p)) if i != gold]
        hardest = max(negative, key=p.__getitem__)
        false_positives = [i for i in chosen if i != gold]
        outcomes[outcome] += 1
        if top != gold: confusion[(candidates[gold], candidates[top])] += 1
        fp_by_class.update(candidates[i] for i in false_positives)
        details.append(dict(id=r['id'], gold=candidates[gold], top=candidates[top],
            outcome=outcome, top1_correct=top == gold, gold_probability=p[gold],
            hardest_negative=candidates[hardest], hardest_negative_probability=p[hardest],
            selected_count=len(chosen), false_positive_count=len(false_positives),
            gold_rank=1+sum(v > p[gold] for v in p),
            positive_nll=-math.log(p[gold]),
            negative_nll=-sum(math.log1p(-p[i]) for i in negative),
            saturated_wrong_negative=sum(p[i] > .99 for i in negative),
            saturated_missed_positive=p[gold] < .01))
    n = len(details)
    return dict(metrics=scorer.result(), outcomes=dict(outcomes),
        top1_correct_but_set_wrong=sum(r['top1_correct'] and r['outcome'] != 'exact' for r in details),
        total_false_positives=sum(r['false_positive_count'] for r in details),
        gold_top3=sum(r['gold_rank'] <= 3 for r in details)/n,
        positive_nll_mean=sum(r['positive_nll'] for r in details)/n,
        negative_nll_sum_mean=sum(r['negative_nll'] for r in details)/n,
        wrong_negative_above_099=sum(r['saturated_wrong_negative'] > 0 for r in details),
        gold_below_001=sum(r['saturated_missed_positive'] for r in details),
        top_confusions=[dict(gold=a,predicted=b,count=v) for (a,b),v in confusion.most_common(20)],
        false_positive_classes=fp_by_class.most_common(20), details=details)


def compact(result):
    return {k: ({a:b for a,b in v.items() if a not in ('classes','confusion','candidate_ids')}
                if k == 'metrics' else v) for k,v in result.items() if k != 'details'}


def run(c):
    root=Path(c['release']); manifest=root/'MANIFEST.json'
    if sha(manifest) != c['release_sha256']: raise ValueError('release identity changed')
    pub=read(manifest); sources={str(manifest):sha(manifest)}
    for name in ('tasks.json','selection.jsonl','learning.jsonl'):
        path=root/name
        if sha(path) != pub['files'][name]: raise ValueError('release file changed: '+name)
        sources[str(path)]=sha(path)
    candidates=[r['id'] for r in read(root/'tasks.json')['intent']['schema']['candidates']]
    labels=[json.loads(s) for s in (root/'selection.jsonl').read_text().splitlines()]
    learning=[json.loads(s) for s in (root/'learning.jsonl').read_text().splitlines()]
    if (any(r['role']!='selection' for r in labels) or
            {r['id'] for r in labels} & {r['id'] for r in learning}):
        raise ValueError('disjoint source training and selection required')
    summaries={}; full={}
    for name, spec in c['points'].items():
        result=read(spec['result']); path=Path(spec['predictions'])
        if (sha(path)!=result['predictions_sha256'] or result['role']!='selection'
                or result['release_sha256']!=c['release_sha256']):
            raise ValueError('prediction publication changed')
        rows=read(path)['records']
        if any(r['policy_id']!=result['policy_id'] for r in rows): raise ValueError('mixed policy')
        diagnosis=diagnose(rows,labels,candidates)
        for key in ('queries','top1_event_accuracy','query_macro_set_f1','exact_set','candidate_binary_nll','candidate_brier'):
            if not math.isclose(diagnosis['metrics'][key],result['metrics'][key],abs_tol=1e-12):
                raise ValueError('independent metric differs: '+key)
        full[name]=diagnosis; summaries[name]=compact(diagnosis)
        sources[str(path)]=sha(path); sources[spec['result']]=sha(spec['result'])
    initial={r['id']:r for r in full['initial']['details']}
    transitions=Counter(initial[r['id']]['outcome']+'->'+r['outcome'] for r in full['final']['details'])
    learned={r['id']:r for r in learning}
    def trace(path):
        data=read(path); pairs=[]
        for q in data['queries']:
            ident=q['request_id'].split('::epoch')[0]
            # Real population only; tail padding is deduplicated below.
            if ident not in learned: raise ValueError('rollout includes non-TRAIN record')
            pairs.append((ident,q['behavior_record'],q['reference_record']))
        return str(path),sha(path),data['update'],pairs
    audit=read(c['training_audit']); transport=read(c['training_transport'])
    if (sha(c['training_transport'])!=audit['transport_sha256'] or
            audit['status']!='stage2_objective_formal_audited'):
        raise ValueError('independently audited TRAIN trajectory required')
    sources[c['training_audit']]=sha(c['training_audit'])
    sources[c['training_transport']]=sha(c['training_transport'])
    train={}
    for window in c['train_windows']:
        paths=[Path(c['training_run'])/f'rollout-{i:04d}.json' for i in range(window['start'],window['stop'])]
        behavior={}; reference={}; updates=[]
        with ThreadPoolExecutor(max_workers=c['io_workers']) as pool:
            for path,digest,update,pairs in pool.map(trace,paths):
                if transport['files'].get('run/'+Path(path).name)!=digest:
                    raise ValueError('TRAIN trace differs from audited transport')
                sources[path]=digest;updates.append(update)
                for ident,b,r in pairs:
                    behavior[ident]=dict(b,id=ident);reference[ident]=dict(r,id=ident)
        ids=sorted(behavior)
        if len(ids)!=window['expected_unique']: raise ValueError('TRAIN window population differs')
        target=[learned[i] for i in ids]
        train[window['name']]=dict(updates=[min(updates),max(updates)],
            distinct_records=len(ids),fixed_endpoint=False,
            behavior=compact(diagnose([behavior[i] for i in ids],target,candidates)),
            reference=compact(diagnose([reference[i] for i in ids],target,candidates)))
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    commit_json(out/'config.json',c)
    commit_json(out/'selection-details.json',{name:value['details'] for name,value in full.items()})
    report=dict(schema='jev-decision-error-diagnosis-v1',selection=summaries,
        transitions=dict(transitions),train=train,sources=sources,
        scope='Selection diagnosis only, never training examples. TRAIN windows contain changing behavior policies, not endpoint metrics. No acoustic/annotation causal attribution without independent evidence.')
    commit_json(out/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,_=load_config(a.config,a.local);run(c['diagnosis'])

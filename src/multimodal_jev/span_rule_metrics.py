"""Paired rule metrics; labels are read only after model predictions exist."""
from collections import defaultdict
from .span_rule_dataset import predicted_fact_rule


def score_rules(rows, predictions):
    expected={r['id']:r for r in rows};actual={r['id']:r for r in predictions}
    if len(expected)!=len(rows) or len(actual)!=len(predictions) or expected.keys()!=actual.keys():
        raise ValueError('complete unique matched predictions required')
    groups=defaultdict(list);pairs=defaultdict(dict);complements=defaultdict(dict)
    for key,row in expected.items():
        pred=actual[key]['predictions'];gold=row['targets'];p=row['provenance'];arm=row['arm']
        if len(pred)!=len(gold) or any(type(v) is not bool for v in pred):raise ValueError('Boolean predictions required')
        executor=None
        if arm!='final_only':
            facts=pred[:2] if arm=='structured' else list(reversed(pred[:2]))
            executor=predicted_fact_rule(p['rule'],dict(zip(p['dependency_keys'],facts,strict=True)))
        scored=dict(final_correct=pred[-1]==gold[-1],all_correct=pred==gold,gold=gold[-1],prediction=pred[-1],
            executor_correct=None if executor is None else executor==gold[-1],
            rule_consistent=None if executor is None else executor==pred[-1])
        groups[(arm,p['rule_id'])].append(scored)
        pairs[(p['recording_id'],p['rule_id'])][arm]=scored
        complements[(p['recording_id'],p['complement_pair_id'],arm)][p['rule_polarity']]=scored
    metrics=[]
    for (arm,rule),items in sorted(groups.items()):
        confusion=[[sum(r['gold']==bool(g) and r['prediction']==bool(p) for r in items) for p in range(2)] for g in range(2)]
        recall=[confusion[g][g]/sum(confusion[g]) if sum(confusion[g]) else None for g in range(2)]
        metrics.append(dict(arm=arm,rule_id=rule,n=len(items),final_accuracy=sum(r['final_correct'] for r in items)/len(items),
            all_span_accuracy=sum(r['all_correct'] for r in items)/len(items),confusion_gold_by_prediction=confusion,
            balanced_accuracy=sum(recall)/2 if None not in recall else None,
            executor_accuracy=None if arm=='final_only' else sum(r['executor_correct'] for r in items)/len(items),
            rule_consistency=None if arm=='final_only' else sum(r['rule_consistent'] for r in items)/len(items)))
    comparisons={}
    for arm in ('structured','reordered'):
        win=loss=same=0
        for group in pairs.values():
            if set(group)!= {'final_only','structured','reordered'}:raise ValueError('missing comparison arm')
            a,b=group[arm]['final_correct'],group['final_only']['final_correct']
            win+=int(a and not b);loss+=int(b and not a);same+=int(a==b)
        comparisons[arm]=dict(wins=win,losses=loss,ties=same,n=win+loss+same)
    inversion={}
    for arm in ('final_only','structured','reordered'):
        items=[v for (r,p,a),v in complements.items() if a==arm]
        if any(set(v)!= {'positive','negative'} for v in items):raise ValueError('missing complementary request')
        inversion[arm]=dict(n=len(items),prediction_flips=sum(v['positive']['prediction']!=v['negative']['prediction'] for v in items),
            both_correct=sum(v['positive']['final_correct'] and v['negative']['final_correct'] for v in items))
    return dict(per_rule=metrics,paired_vs_final_only=comparisons,complement_response=inversion,
        scope='Learning-only paired diagnostic; requests share audio and rules, not independent observations; no held-out or causal CoT claim')

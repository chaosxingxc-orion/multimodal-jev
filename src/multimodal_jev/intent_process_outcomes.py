"""Separate correct process, correct Final, and self-consistent wrong answers.

Consumes semantic records from authenticated intent scoring, not model prose.
These are descriptive learning diagnostics, not causal process attribution.
"""
from collections import Counter
import itertools
import math


def checked(records):
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('nonempty distinct semantic records required')
    result = {}
    for row in records:
        gold, pred, probabilities = row['gold'], row['prediction'], row['gold_probability']
        if (len(gold) != 3 or len(pred) != 3 or len(probabilities) != 3
                or any(not isinstance(x, str) or not x for x in gold+pred)
                or any(type(p) not in (int,float) or not math.isfinite(p) or not 0<=p<=1 for p in probabilities)):
            raise ValueError('three finite canonical scenario/action/Final decisions required')
        if gold[0]+'_'+gold[1] != gold[2]:
            raise ValueError('reviewed canonical process projection differs')
        correct = tuple(a == b for a,b in zip(gold,pred,strict=True))
        compatible = pred[0]+'_'+pred[1] == pred[2]
        if (type(row['correct']) is not bool or row['correct'] != correct[-1]
                or type(row['compatible']) is not bool or row['compatible'] != compatible):
            raise ValueError('derived correctness or compatibility differs')
        result[row['id']] = dict(gold=gold, signature=''.join(str(int(x)) for x in correct),
            process_correct=all(correct[:-1]), final_correct=correct[-1], compatible=compatible,
            expected_process_reward=.1*sum(probabilities[:-1]), expected_final_reward=probabilities[-1])
    return result


def summarize(records):
    rows=list(checked(records).values()); n=len(rows)
    patterns={''.join(s):0 for s in itertools.product('01',repeat=3)}
    patterns.update(Counter(r['signature'] for r in rows))
    conditional={}
    for name,value in (('both_process_correct',True),('some_process_wrong',False)):
        subset=[r for r in rows if r['process_correct']==value]
        k=sum(r['final_correct'] for r in subset)
        conditional[name]=dict(recordings=len(subset),final_correct=k,final_accuracy=k/len(subset) if subset else None)
    return dict(recordings=n,signature_order=['scenario','action','Final'],correctness_patterns=patterns,
        all_three_correct=patterns['111'],both_process_correct=sum(r['process_correct'] for r in rows),
        final_correct=sum(r['final_correct'] for r in rows),
        compatible=sum(r['compatible'] for r in rows),
        compatible_but_final_wrong=sum(r['compatible'] and not r['final_correct'] for r in rows),
        both_process_correct_but_final_wrong=patterns['110'],
        final_correct_with_process_error=sum(v for k,v in patterns.items() if k[-1]=='1' and k!='111'),
        conditional_final=conditional,
        mean_expected_process_reward=sum(r['expected_process_reward'] for r in rows)/n,
        mean_expected_final_reward=sum(r['expected_final_reward'] for r in rows)/n,
        scope='Unweighted learning population. Process reward is counterfactual 0.1/0.1; no training-arm inference or causal mediation claim.')


def transitions(before,after):
    left,right=checked(before),checked(after)
    if left.keys()!=right.keys() or any(left[i]['gold']!=right[i]['gold'] for i in left):
        raise ValueError('identical paired sources and gold required')
    cells=Counter((left[i]['signature'],right[i]['signature']) for i in left)
    process_only=final_gain=final_loss=both_gain=0
    for uid,a in left.items():
        b=right[uid]; pg=not a['process_correct'] and b['process_correct']
        fg=not a['final_correct'] and b['final_correct']
        process_only+=pg and not a['final_correct'] and not b['final_correct']
        final_gain+=fg;final_loss+=a['final_correct'] and not b['final_correct'];both_gain+=pg and fg
    return dict(recordings=len(left),final_gains=final_gain,final_losses=final_loss,
        both_process_repaired_final_still_wrong=process_only,both_process_and_final_repaired=both_gain,
        cells=[dict(before=a,after=b,recordings=n) for (a,b),n in sorted(cells.items())])

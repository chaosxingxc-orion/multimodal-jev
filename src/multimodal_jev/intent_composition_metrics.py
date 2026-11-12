"""Exploratory ontology slices on complete accepted learning predictions."""
import math
from .composition_support import audit


def compare(before,after,pairs):
    structure=audit(pairs)['minimum_identifying_components']
    def indexed(rows):
        result={r['id']:r for r in rows}
        if not rows or len(result)!=len(rows):raise ValueError('complete unique records required')
        if len({len(r['gold']) for r in rows})!=1:raise ValueError('one fixed template required')
        for row in rows:
            gold=row['gold'];prediction=row['prediction'];prob=row['gold_probability']
            if len(gold) not in (1,3) or len(prediction)!=len(gold) or len(prob)!=len(gold):
                raise ValueError('canonical D/Q spans required')
            if gold[-1] not in pairs or prediction[-1] not in pairs:raise ValueError('full known ontology required')
            if len(gold)==3 and tuple(gold[:2])!=tuple(pairs[gold[-1]]):raise ValueError('gold projection differs')
            if any(type(p) not in (float,int) or not math.isfinite(p) or not 0<=p<=1 for p in prob):
                raise ValueError('valid gold probabilities required')
            if type(row['correct']) is not bool or row['correct']!=(gold[-1]==prediction[-1]):
                raise ValueError('correctness differs')
        return result
    left,right=indexed(before),indexed(after)
    if set(left)!=set(right):raise ValueError('exact paired population required')
    if any(left[k]['gold']!=right[k]['gold'] for k in left):raise ValueError('paired labels differ')
    def metrics(rows):
        n=len(rows)
        result=dict(recordings=n,correct=sum(r['correct'] for r in rows),
            final_accuracy=sum(r['correct'] for r in rows)/n if n else None,
            mean_gold_probability=sum(r['gold_probability'][-1] for r in rows)/n if n else None,
            predicted_classes=len({r['prediction'][-1] for r in rows}))
        if rows and len(rows[0]['gold'])==3:
            both=[r for r in rows if r['prediction'][:2]==r['gold'][:2]]
            consistent=[r for r in rows if tuple(r['prediction'][:2])==tuple(pairs[r['prediction'][-1]])]
            result.update(process_both_correct=len(both),final_correct_given_process_both_correct=
                sum(r['correct'] for r in both)/len(both) if both else None,
                compatible=len(consistent),compatible_but_final_wrong=sum(not r['correct'] for r in consistent))
        return result
    result={}
    for name,ids in [('all',sorted(left)),('single_component',sorted(k for k,r in left.items() if structure[r['gold'][-1]]==1)),
                     ('joint_components',sorted(k for k,r in left.items() if structure[r['gold'][-1]]==2))]:
        a,b=[left[k] for k in ids],[right[k] for k in ids]
        result[name]=dict(before=metrics(a),after=metrics(b),
            repaired=[k for k in ids if not left[k]['correct'] and right[k]['correct']],
            regressed=[k for k in ids if left[k]['correct'] and not right[k]['correct']],
            probability_improved=sum(right[k]['gold_probability'][-1]>left[k]['gold_probability'][-1] for k in ids))
    return result

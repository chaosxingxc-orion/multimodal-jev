"""Paired learning-only change, with the actual update batch separated."""
import math


def compare(before,after,selected_ids):
    selected=list(selected_ids)
    if not selected or len(set(selected))!=len(selected):
        raise ValueError('distinct actual selected recordings required')
    if before['role']!='learning' or after['role']!='learning' or before['base_id']!=after['base_id']:
        raise ValueError('same-base learning reports required')
    reports={};population=None
    for arm in ('D','Q'):
        old=before['reports'][arm];new=after['reports'][arm]
        if old['admission']!=new['admission']:raise ValueError('population admission changed')
        left=old['metrics']['records'];right=new['metrics']['records']
        a={r['id']:r for r in left};b={r['id']:r for r in right}
        if len(a)!=len(left) or len(b)!=len(right) or a.keys()!=b.keys():
            raise ValueError('complete distinct paired population required')
        if not set(selected)<=a.keys() or (population is not None and population!=set(a)):
            raise ValueError('selected or cross-template population differs')
        population=set(a)
        if any(a[k]['gold']!=b[k]['gold'] for k in a):raise ValueError('paired gold changed')
        groups={}
        for name,ids in (('all_learning',sorted(a)),('updated_this_step',selected),
                         ('other_learning',sorted(set(a)-set(selected)))):
            if not ids:continue
            def metrics(rows):
                probabilities=[rows[k]['gold_probability'][-1] for k in ids]
                if any(not math.isfinite(p) or not 0<p<=1 for p in probabilities):
                    raise ValueError('finite positive gold probabilities required')
                return dict(final_accuracy=sum(rows[k]['correct'] for k in ids)/len(ids),
                    final_expected_reward=sum(probabilities)/len(ids),
                    final_nll=-sum(math.log(p) for p in probabilities)/len(ids),
                    predicted_classes=len({rows[k]['prediction'][-1] for k in ids}))
            prior=metrics(a);later=metrics(b)
            groups[name]=dict(recordings=len(ids),ids=ids,before=prior,after=later,
                delta={k:later[k]-prior[k] for k in prior},
                wrong_to_right=sum(not a[k]['correct'] and b[k]['correct'] for k in ids),
                right_to_wrong=sum(a[k]['correct'] and not b[k]['correct'] for k in ids),
                improved_gold_probability=sum(b[k]['gold_probability'][-1]>a[k]['gold_probability'][-1] for k in ids))
        reports[arm]=groups
    return dict(base_id=before['base_id'],before_adapter_id=before['adapter_id'],after_adapter_id=after['adapter_id'],
        reports=reports,role='learning',scope='Paired same-population learning diagnostics; other_learning is not development/test or evidence of held-out gain.')

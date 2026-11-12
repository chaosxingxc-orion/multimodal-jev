"""Complete-prefix learning curves; never infer held-out gain or select best."""
from collections import Counter
import math


def summarize(points,schedule,total_updates):
    if type(total_updates)!=int or total_updates<1 or len(schedule)!=total_updates:
        raise ValueError('fixed full update schedule required')
    if not points or [p['update'] for p in points]!=list(range(len(points))) or len(points)>total_updates+1:
        raise ValueError('contiguous prefix including step0 required')
    initial=points[0]['score'];base=initial['base_id'];population={}
    for arm in ('D','Q'):
        rows=initial['reports'][arm]['metrics']['records']
        population[arm]={r['id']:r['gold'] for r in rows}
        if len(population[arm])!=len(rows):raise ValueError('distinct fixed population required')
    if population['D'].keys()!=population['Q'].keys():raise ValueError('template populations differ')
    for batch in schedule:
        if not batch or len(batch)!=len(set(batch)) or not set(batch)<=population['D'].keys():
            raise ValueError('distinct scheduled learning sources required')
    exposures=Counter();curve=[]
    for point in points:
        step=point['update'];report=point['score']
        if report['role']!='learning' or report['base_id']!=base:raise ValueError('same-base learning scores required')
        if step:exposures.update(schedule[step-1])
        arms={}
        for arm in ('D','Q'):
            payload=report['reports'][arm];m=payload['metrics'];rows=m['records']
            if (payload['admission']!=initial['reports'][arm]['admission'] or len(rows)!=len(population[arm])
                    or {r['id']:r['gold'] for r in rows}!=population[arm]):
                raise ValueError('population/labels/admission changed along curve')
            groups={}
            for name,subset in (('all_learning',rows),('ever_updated',[r for r in rows if r['id'] in exposures]),
                                ('not_yet_updated',[r for r in rows if r['id'] not in exposures])):
                if not subset:groups[name]=dict(recordings=0,final_accuracy=None,final_expected_reward=None,final_nll=None);continue
                probs=[r['gold_probability'][-1] for r in subset]
                if any(not math.isfinite(p) or not 0<p<=1 for p in probs):raise ValueError('finite nonzero gold probabilities required')
                groups[name]=dict(recordings=len(subset),final_accuracy=sum(r['correct'] for r in subset)/len(subset),
                    final_expected_reward=sum(probs)/len(subset),final_nll=-sum(math.log(p) for p in probs)/len(subset))
            arms[arm]=dict(groups=groups,final_macro_f1_observed=m['final_macro_f1_observed'],
                final_predicted_classes=m['final_predicted_classes'],final_dominant_prediction_fraction=m['final_dominant_prediction_fraction'])
        curve.append(dict(update=step,adapter_id=report['adapter_id'],sample_uses=sum(exposures.values()),
            distinct_updated_sources=len(exposures),exposure_counts=dict(sorted(Counter(exposures.values()).items())),arms=arms))
    changes={}
    for arm in ('D','Q'):
        values=[p['arms'][arm]['groups']['all_learning']['final_accuracy'] for p in curve]
        changes[arm]=dict(last_minus_initial=values[-1]-values[0],adjacent_decreases=sum(b<a for a,b in zip(values,values[1:])),
            normalized_accuracy_auc=sum((a+b)/2 for a,b in zip(values,values[1:]))/(len(values)-1) if len(values)>1 else None)
    return dict(total_planned_updates=total_updates,accepted_updates=points[-1]['update'],
        complete=points[-1]['update']==total_updates,curve=curve,accuracy_summary=changes,
        scope='Learning-only complete-prefix curve. Ever/not-yet updated subsets change with the fixed schedule and are not held-out evaluations. No best-checkpoint selection.')

"""Full-candidate update diagnostics on a fixed, externally admitted population.

Margins use gold minus the strongest wrong candidate, not top1 minus top2.
This module never selects a policy or turns learning sources into held-out data.
"""
import math
from .span_policy_drift import diagnose, distribution


def margins(logp, target):
    values = distribution(logp)
    if type(target) is not int or not 0 <= target < len(values):
        raise ValueError('in-support integer gold required')
    rival = max(v for i,v in enumerate(values) if i != target)
    return dict(gold_log_margin=values[target]-rival,
                gold_probability_margin=math.exp(values[target])-math.exp(rival))


def summarize(rows, selected):
    """rows contain id, weight, target and old/new complete Final logprobs."""
    ids = [r['id'] for r in rows]
    if (not ids or len(set(ids)) != len(ids) or not selected
            or len(set(selected)) != len(selected) or not set(selected) <= set(ids)):
        raise ValueError('distinct complete population and selected sources required')
    values = []
    for row in rows:
        weight = row['weight']
        if type(weight) not in (int,float) or not math.isfinite(weight) or weight <= 0:
            raise ValueError('positive finite source weights required')
        old,new,target = row['old'],row['new'],row['target']
        drift = diagnose(old,new,target)
        before,after = margins(old,target),margins(new,target)
        correct = [max(range(len(p)),key=p.__getitem__)==target for p in (old,new)]
        values.append(dict(id=row['id'],weight=weight,before=before,after=after,
            before_correct=correct[0],after_correct=correct[1],drift=drift))
    groups = {}
    chosen = set(selected)
    for name,subset in (('all_learning',values),
                        ('updated_this_step',[v for v in values if v['id'] in chosen]),
                        ('other_learning',[v for v in values if v['id'] not in chosen])):
        if not subset:
            groups[name] = dict(recordings=0,metrics=None)
            continue
        total = sum(v['weight'] for v in subset)
        measures = {}
        for phase in ('before','after'):
            measures[phase] = {k:sum(v[phase][k] for v in subset)/len(subset)
                              for k in subset[0][phase]}
            measures[phase]['correct'] = sum(v[phase+'_correct'] for v in subset)
            measures[phase]['nonpositive_gold_margin'] = sum(v[phase]['gold_log_margin'] <= 0 for v in subset)
        keys = ('kl_old_to_new','kl_new_to_old','total_variation','old_mass_outside_clip',
                'old_mass_clipped_by_advantage','gold_probability_delta')
        measures['mean_drift'] = {k:sum(v['drift'][k] for v in subset)/len(subset) for k in keys}
        measures['weighted_drift'] = {k:sum(v['weight']*v['drift'][k] for v in subset)/total for k in keys}
        measures['max_kl_old_to_new'] = max(v['drift']['kl_old_to_new'] for v in subset)
        measures['wrong_to_right'] = sum(not v['before_correct'] and v['after_correct'] for v in subset)
        measures['right_to_wrong'] = sum(v['before_correct'] and not v['after_correct'] for v in subset)
        groups[name] = dict(recordings=len(subset),metrics=measures)
    return dict(groups=groups,records=values,clip_ratio=.2,
        scope='Final marginal learning diagnostics. Clip mass is recomputed after the update with correctness advantage signs; '
              'not the training logger clip fraction, an objective-specific surrogate or independent efficacy.')


def training_metrics(results):
    """One synchronized update, distinguish pre-update logs from post-update drift."""
    if len(results) != 4 or [r['rank'] for r in results] != list(range(4)):
        raise ValueError('complete ordered four-rank results required')
    updates = {r['updates'] for r in results}
    adapters = {r['adapter_id'] for r in results}
    if len(updates) != 1 or len(adapters) != 1:
        raise ValueError('same update and exported policy required')
    norms = []
    traces = []
    for row in results:
        metrics = row['metrics']; norm = metrics['actor/grad_norm']
        if len(norm) != 1 or type(norm[0]) not in (int,float) or not math.isfinite(norm[0]) or norm[0] < 0:
            raise ValueError('one finite global gradient norm required')
        norms.append(norm[0])
        fractions = metrics['actor/pg_clipfrac']
        if not fractions or any(type(x) not in (int,float) or not math.isfinite(x) or not 0 <= x <= 1 for x in fractions):
            raise ValueError('finite pre-update microbatch clipping fractions required')
        traces.append(dict(rank=row['rank'],preupdate_microbatch_clipfractions=fractions))
    if any(not math.isclose(v,norms[0],rel_tol=1e-6,abs_tol=1e-8) for v in norms):
        raise ValueError('synchronized global gradient norms disagree')
    return dict(updates=next(iter(updates)),adapter_id=next(iter(adapters)),
        global_grad_norm_before_clip=norms[0],grad_clip_limit=1.,
        gradient_clip_triggered=norms[0]>1.,ranks=traces,
        scope='Training-reported pre-update metrics at fixed grad_clip=1; '
              'does not measure parameter displacement or constrain post-update KL.')

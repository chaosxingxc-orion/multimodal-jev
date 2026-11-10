"""Describe reward/advantage coverage without selecting a training algorithm."""
from collections import defaultdict
import math


def summarize_signal(audit, examples, *, threshold=1e-5):
    if type(threshold) not in (int,float) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError('positive finite descriptive threshold required')
    lookup={ex.example_id:ex for ex in examples}
    if not lookup or len(lookup)!=len(examples):
        raise ValueError('nonempty unique example selection required')
    if any(ex.split!='train' for ex in examples):
        raise ValueError('TRAIN-only signal diagnosis')
    for ex in examples:
        if not ex.metadata.get('source_group') or not ex.metadata.get('variant'):
            raise ValueError('source group and variant required')
    rows=audit['per_response'];bonuses=audit['pair_bonuses']
    centered=audit['center_only_control']
    n=audit['replicates']
    if type(n) is not int or n<2 or audit['status']!='passed':
        raise ValueError('successful grouped advantage audit required')
    if (len(rows)!=audit['responses'] or len(rows)!=len(lookup)*n or
            any(len(values)!=len(rows) for values in (bonuses,centered['grpo'],centered['pair_grpo']))):
        raise ValueError('incomplete aligned advantage arrays')
    seen=set();by_view=defaultdict(list);by_group=defaultdict(list);all_rows=[]
    for i,row in enumerate(rows):
        uid,replicate=row['example_id'],row['replicate']
        if uid not in lookup or type(replicate) is not int or not 0<=replicate<n or (uid,replicate) in seen:
            raise ValueError('unknown or duplicate response identity')
        seen.add((uid,replicate));ex=lookup[uid]
        values=[bonuses[i],row['grpo'],row['pair_grpo'],centered['grpo'][i],centered['pair_grpo'][i]]
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
            raise ValueError('finite numeric rewards/advantages required')
        record={'uid':uid,'group':ex.metadata['source_group'],'bonus':bonuses[i],
                'standard_delta':abs(row['pair_grpo']-row['grpo']),
                'center_delta':abs(centered['pair_grpo'][i]-centered['grpo'][i]),
                'standard_sign_change':(row['grpo']>0)-(row['grpo']<0)!=(row['pair_grpo']>0)-(row['pair_grpo']<0)}
        all_rows.append(record);by_view[ex.metadata['variant']].append(record)
        by_group[ex.metadata['source_group']].append(record)

    def aggregate(items):
        result={'responses':len(items),'prompts':len({r['uid'] for r in items}),
                'recordings':len({r['group'] for r in items}),
                'nonzero_bonus_responses':sum(r['bonus']!=0 for r in items),
                'standard_sign_change_responses':sum(r['standard_sign_change'] for r in items)}
        for label,field in [('standard','standard_delta'),('center_only','center_delta')]:
            material=[r for r in items if r[field]>threshold]
            result[label]={'max_abs_difference':max(r[field] for r in items),
                           'material_responses':len(material),
                           'material_prompts':len({r['uid'] for r in material}),
                           'material_recordings':len({r['group'] for r in material})}
        return result
    return {'threshold':threshold,'threshold_rule':'absolute difference strictly greater than threshold',
            'overall':aggregate(all_rows),
            'by_variant':{k:aggregate(v) for k,v in sorted(by_view.items())},
            'by_recording':{k:aggregate(v) for k,v in sorted(by_group.items())},
            'scope':'TRAIN mechanism coverage; response-level signs, not token counts; '
                    'no actor update, winning normalization or efficacy inference'}


def summarize_sequence_coefficients(audit, examples, records, *, threshold=1e-5):
    """Sum advantage differences for identical prompt/response token sequences.

    This is a coefficient diagnostic, not a model gradient calculation. Equal
    sequences share derivatives only with deterministic forward and identical
    inputs; PPO ratios/clipping also depend on stored old log probabilities.
    """
    summarize_signal(audit, examples, threshold=threshold)
    rows = audit['per_response']
    if len(records) != len(rows):
        raise ValueError('complete aligned rollout required')
    lookup = {ex.example_id: ex for ex in examples}
    buckets = defaultdict(list)
    for i, (row, record) in enumerate(zip(rows, records)):
        if (row['example_id'], row['replicate']) != (record['example_id'], record['replicate']):
            raise ValueError('rollout response order differs from audit')
        tokens, logps = record['token_ids'], record['log_probs']
        if (not tokens or len(tokens) != len(logps)
                or any(type(t) is not int or t < 0 for t in tokens)
                or any(type(p) not in (int, float) or not math.isfinite(p) for p in logps)):
            raise ValueError('valid response tokens and finite aligned log probabilities required')
        buckets[(row['example_id'], tuple(tokens))].append(i)
    projected = []
    for (uid, tokens), indices in buckets.items():
        ex = lookup[uid]
        item = {'example_id': uid, 'source_group': ex.metadata['source_group'],
                'variant': ex.metadata['variant'], 'replicates': [rows[i]['replicate'] for i in indices],
                'response_tokens': len(tokens),
                'max_old_logprob_spread': max(
                    max(records[i]['log_probs'][t] for i in indices)
                    - min(records[i]['log_probs'][t] for i in indices)
                    for t in range(len(tokens)))}
        for mode in ('standard', 'center_only'):
            delta = [(rows[i]['pair_grpo'] - rows[i]['grpo']) if mode == 'standard'
                     else (audit['center_only_control']['pair_grpo'][i]
                           - audit['center_only_control']['grpo'][i]) for i in indices]
            item[mode] = {'signed_coefficient_sum': math.fsum(delta),
                          'max_response_difference': max(abs(d) for d in delta)}
        projected.append(item)

    def aggregate(items):
        result = {'sequence_buckets': len(items),
                  'max_old_logprob_spread': max(r['max_old_logprob_spread'] for r in items)}
        for mode in ('standard', 'center_only'):
            material = [r for r in items if abs(r[mode]['signed_coefficient_sum']) > threshold]
            cancelled = [r for r in items if r[mode]['max_response_difference'] > threshold
                         and abs(r[mode]['signed_coefficient_sum']) <= threshold]
            result[mode] = {'material_buckets': len(material),
                            'material_prompts': len({r['example_id'] for r in material}),
                            'material_recordings': len({r['source_group'] for r in material}),
                            'cancelled_material_buckets': len(cancelled),
                            'max_abs_coefficient_sum': max(abs(r[mode]['signed_coefficient_sum']) for r in items)}
        return result

    variants = sorted({r['variant'] for r in projected})
    return {'threshold': threshold, 'threshold_rule': 'absolute unnormalized signed coefficient sum strictly greater than threshold',
            'overall': aggregate(projected),
            'by_variant': {v: aggregate([r for r in projected if r['variant'] == v]) for v in variants},
            'per_sequence': projected,
            'scope': 'identical example_id and complete response token IDs only; signed advantage sums before loss normalization; '
                     'conditional deterministic-gradient cancellation diagnostic, not measured gradients or efficacy; '
                     'nonzero old-logprob spread, stochastic forward or different PPO clipping can prevent cancellation'}

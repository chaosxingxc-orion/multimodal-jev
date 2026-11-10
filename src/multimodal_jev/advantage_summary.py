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

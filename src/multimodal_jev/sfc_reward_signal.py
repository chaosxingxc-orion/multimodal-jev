"""Describe complete, validated reward batches without inventing RL efficacy."""
from collections import defaultdict
import math


def summarize(records, batch):
    n=len(records)
    if (not n or any(len(batch[k])!=n for k in ('rewards','uid','components'))
            or type(batch['group_size']) is not int or batch['group_size']<2):
        raise ValueError('aligned complete reward batch required')
    groups=defaultdict(list);types=defaultdict(lambda:dict(responses=0,correct=0,format_invalid=0,
        schema_invalid=0,length_limited=0,response_tokens=0,reward_sum=0.))
    for index,(record,reward,uid,component) in enumerate(zip(records,batch['rewards'],batch['uid'],batch['components'],strict=True)):
        if type(reward) not in (int,float) or not math.isfinite(reward):raise ValueError('finite rewards required')
        groups[uid].append(index)
        kind=record['decision_id'].rsplit('.',1)[-1]
        if kind not in ('present','absent','calls'):raise ValueError('declared SFC decision type required')
        if component['decision_id']!=record['decision_id']:raise ValueError('reward component alignment differs')
        item=types[kind];item['responses']+=1;item['correct']+=int(component['semantic']==1)
        item['format_invalid']+=int(not component['format_valid'])
        item['schema_invalid']+=int(component['format_valid'] and not component['schema_valid'])
        item['length_limited']+=int(record['finish_reason']=='length')
        item['response_tokens']+=len(record['token_ids']);item['reward_sum']+=reward
    counts=dict(contrast_groups=0,equal_reward_all_correct=0,equal_reward_all_wrong=0,
                equal_reward_mixed_correctness=0,identical_text_groups=0,identical_token_groups=0)
    ranges=[]
    for indices in groups.values():
        if (len(indices)!=batch['group_size'] or {records[i]['replicate'] for i in indices}!=set(range(batch['group_size']))
                or len({(records[i]['example_id'],records[i]['decision_id']) for i in indices})!=1):
            raise ValueError('complete same-input K group required')
        rewards=[batch['rewards'][i] for i in indices];span=max(rewards)-min(rewards);ranges.append(span)
        correct=sum(batch['components'][i]['semantic']==1 for i in indices)
        if span>0:counts['contrast_groups']+=1
        else:
            key='all_correct' if correct==len(indices) else 'all_wrong' if correct==0 else 'mixed_correctness'
            counts['equal_reward_'+key]+=1
        counts['identical_text_groups']+=int(len({records[i]['raw_text'] for i in indices})==1)
        counts['identical_token_groups']+=int(len({tuple(records[i]['token_ids']) for i in indices})==1)
    if len(groups)!=batch['prompt_groups']:raise ValueError('complete prompt group count required')
    summaries=batch['replicate_summaries']
    if len(summaries)!=batch['bundles']*batch['group_size']:raise ValueError('complete bundle replicates required')
    return dict(responses=n,prompt_groups=len(groups),group_size=batch['group_size'],mode=batch['mode'],
        groups=counts,reward_range_mean=math.fsum(ranges)/len(ranges),reward_range_max=max(ranges),
        decision_types=dict(types),bundle_replicates=len(summaries),
        native_calls_exact=sum(s['native_calls_exact'] for s in summaries),
        all_decisions_correct=sum(s['all_decisions_correct'] for s in summaries),
        scope='Descriptive complete sampled reward groups. Equal outcome rewards give zero centered/standardized '
              'GRPO advantage; contrast count is not policy gradient, learning efficacy or generalization.')

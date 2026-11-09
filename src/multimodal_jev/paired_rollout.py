"""Pair-GRPO reward composition with explicit replicate alignment.

A malformed response is never interpreted as the semantic null decision.
Only the pair bonus is added to the shared strict single-decision reward.
"""
from __future__ import annotations
from collections import defaultdict
import json
from .answers import parse_answer,AnswerError
from .reward import compute_score
from .pair_reward import paired_rewards


def compose_rewards(examples,records,mode='grpo',bonus_weight=0.2):
    if len(examples)!=len(records):raise ValueError('example/record alignment required')
    if mode not in {'grpo','pair-grpo'}:raise ValueError('unsupported reward mode')
    rewards=[];values=[];replicates=defaultdict(list)
    for index,(ex,record) in enumerate(zip(examples,records)):
        if ex.example_id!=record.example_id:raise ValueError('reward example identity mismatch')
        rewards.append(compute_score('jev',record.text,json.dumps(ex.to_record())))
        try:values.append(parse_answer(record.text)['value'])
        except AnswerError:values.append(object()) # invalid, never semantic null
        replicates[record.replicate].append(index)
    diagnostics={'mode':mode,'base_rewards':rewards.copy(),'pair_bonuses':[0.0]*len(rewards)}
    if mode=='pair-grpo':
        relations={}
        for replicate,indices in replicates.items():
            scored,stats=paired_rewards([examples[i] for i in indices],[values[i] for i in indices],bonus_weight)
            relations[str(replicate)]=stats
            for i,row in zip(indices,scored):
                bonus=bonus_weight*row['pair_bonus']
                rewards[i]+=bonus;diagnostics['pair_bonuses'][i]=bonus
        diagnostics['relations_by_replicate']=relations
    return rewards,diagnostics

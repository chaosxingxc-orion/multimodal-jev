#!/usr/bin/env python3
"""Read-only comparison of actual rollout rewards and pinned verl advantages."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.rl_run import load_rollout_run
from multimodal_jev.paired_rollout import compose_rewards


def run(rollout):
    identity,examples,records=load_rollout_run(rollout)
    lookup={ex.example_id:ex for ex in examples}
    ordered=[lookup[r.example_id] for r in records]
    rewards={};advantages={};details={}
    width=max(len(r.token_ids) for r in records)
    mask=torch.zeros(len(records),width)
    for i,r in enumerate(records):mask[i,:len(r.token_ids)]=1
    uid=np.array([r.example_id for r in records],dtype=object)
    for mode in ('grpo','pair-grpo'):
        rewards[mode],details[mode]=compose_rewards(ordered,records,mode,.2)
        token_rewards=torch.zeros_like(mask)
        for i,r in enumerate(records):token_rewards[i,len(r.token_ids)-1]=rewards[mode][i]
        adv,_=compute_grpo_outcome_advantage(token_rewards,mask,uid,norm_adv_by_std_in_grpo=True)
        assert torch.isfinite(adv).all()
        advantages[mode]=adv
    difference=(advantages['grpo']-advantages['pair-grpo']).abs()
    centered={}
    for mode in ('grpo','pair-grpo'):
        token_rewards=torch.zeros_like(mask)
        for i,r in enumerate(records):token_rewards[i,len(r.token_ids)-1]=rewards[mode][i]
        centered[mode],_=compute_grpo_outcome_advantage(token_rewards,mask,uid,norm_adv_by_std_in_grpo=False)
        assert torch.isfinite(centered[mode]).all()
    centered_difference=(centered['grpo']-centered['pair-grpo']).abs()
    return {'status':'passed','policy_id':identity['policy_id'],
        'rollout_sha256':file_hash(rollout/'rollouts.jsonl'),'responses':len(records),
        'replicates':identity['config']['sampling']['n'],'rewards':rewards,
        'pair_bonuses':details['pair-grpo']['pair_bonuses'],
        'nonzero_bonus_responses':sum(v!=0 for v in details['pair-grpo']['pair_bonuses']),
        'max_advantage_difference':float(difference[mask.bool()].max()),
        'advantage_sign_changes':int(((advantages['grpo'].sign()!=advantages['pair-grpo'].sign())&mask.bool()).sum()),
        'center_only_control':{'max_advantage_difference':float(centered_difference[mask.bool()].max()),
            'grpo':[float(row[0]) for row in centered['grpo']],
            'pair_grpo':[float(row[0]) for row in centered['pair-grpo']]},
        'per_response':[{'example_id':r.example_id,'replicate':r.replicate,
            'grpo':float(advantages['grpo'][i,0]),'pair_grpo':float(advantages['pair-grpo'][i,0])}
            for i,r in enumerate(records)],
        'scope':'actual rollout reward/advantage diagnosis, no model update or algorithm efficacy'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rollout',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2)
    atomic_json(args.output,run(args.rollout))

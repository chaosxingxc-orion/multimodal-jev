#!/usr/bin/env python3
"""Test whether learning-frequency reward weights survive actual GRPO normalization."""
import argparse
import inspect
import json
from pathlib import Path

from multimodal_jev.baseline import atomic_json,file_hash


def run(audit_path,audit_hash):
    if file_hash(audit_path)!=audit_hash:raise ValueError('fixed full learning audit changed')
    audit=json.loads(audit_path.read_text());counts=audit['counts']
    if (audit['schema']!='sfc-jev-complete-learning-reward-audit-v1'
            or audit['partition']['part']!='learning' or audit['model_called'] is not False
            or audit['source_groups']!=3600 or counts['bundles']!=4428
            or counts['functions']!=23070 or counts['active_functions']!=6085
            or counts['inactive_functions']!=16985 or counts['decisions']!=69210):
        raise ValueError('registered complete learning distribution required')
    import numpy as np
    import torch
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    torch.set_num_threads(2)
    # Each prompt has a fixed gold activity class across all K=4 responses.
    # Include mixed success, distinct wrong/format penalties, and an equal group.
    reward=torch.tensor([[1.,0.,-1.,1.],[1.,0.,-1.,1.],[1.2,1.,-.1,-1.],[0.,0.,0.,0.]])
    activity=['active','inactive','active','inactive']
    weights={k:counts['functions']/(2*counts[k+'_functions']) for k in ('active','inactive')}
    rowweights=torch.tensor([weights[k] for k in activity]).repeat_interleave(4)
    uid=np.repeat(np.arange(4),4)
    mask=torch.tensor([[1.,1.,0.],[1.,1.,1.],[1.,0.,0.],[1.,1.,0.]]).repeat(4,1)
    lengths=mask.sum(-1).long();base=torch.zeros_like(mask)
    base[torch.arange(16),lengths-1]=reward.flatten()
    reports={}
    for normalized in (False,True):
        before,_=compute_grpo_outcome_advantage(base,mask,uid,norm_adv_by_std_in_grpo=normalized)
        weighted,_=compute_grpo_outcome_advantage(base*rowweights[:,None],mask,uid,norm_adv_by_std_in_grpo=normalized)
        post=before*rowweights[:,None]
        if normalized:
            # A common positive multiplier cancels against each prompt's std,
            # except epsilon. This is the failure mode this probe must expose.
            torch.testing.assert_close(weighted,before,rtol=0,atol=1e-5)
            assert float((post-before).abs().max())>.1
        else:
            torch.testing.assert_close(weighted,post,rtol=0,atol=1e-6)
        assert not weighted[mask==0].any() and not weighted[12:].any()
        reports['standardized' if normalized else 'centered']=dict(
            base_first_token=before[:,0].reshape(4,4).tolist(),
            pre_normalization_weighted_first_token=weighted[:,0].reshape(4,4).tolist(),
            post_normalization_weighted_first_token=post[:,0].reshape(4,4).tolist(),
            pre_weight_max_difference_from_base=float((weighted-before).abs().max()),
            pre_vs_post_max_difference=float((weighted-post).abs().max()))
    # Exact class-mass calculation from the full learning audit, not new labels.
    raw_empty=counts['inactive_functions']/counts['functions']
    weighted_empty=weights['inactive']*counts['inactive_functions']/sum(
        weights[k]*counts[k+'_functions'] for k in weights)
    assert abs(weighted_empty-.5)<1e-12
    return dict(schema='sfc-jev-grpo-reward-balance-diagnostic-v1',audit_sha256=audit_hash,
        learning_counts=counts,class_weights=weights,
        synthetic_empty_raw_reward_fraction=raw_empty,
        synthetic_empty_weighted_reward_fraction=weighted_empty,
        reports=reports,core_source_sha256=file_hash(inspect.getfile(compute_grpo_outcome_advantage)),
        model_loaded=False,weight_updates=0,
        scope='Fixed learning-label counts plus synthetic K=4 reward groups evaluated with real verl. '
              'Reward rescaling before same-prompt standardization cancels; post-standardization weights '
              'or centering change numerical advantages. No actor gradients, efficacy or reward-policy change.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--audit-hash',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve prior probe')
    result=run(a.audit,a.audit_hash);atomic_json(a.output,result);print(json.dumps(result,indent=2))

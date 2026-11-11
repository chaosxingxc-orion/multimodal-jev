#!/usr/bin/env python3
"""Measure sampling credit noise from authenticated real rule behavior policies."""
import argparse
import json
from pathlib import Path
import torch
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_credit_diagnostic import exact_additive_logit_gradient,sampled_logit_gradient
from multimodal_jev.span_rule_objectives import expected_rule_reward,rule_sample_rewards
from multimodal_jev.span_policy import sample_spans
from infer_span_rule_baseline import select_rows
from verify_span_rule_step import verify


def diagnose(root,repeats=256):
    reports=[]
    for branch in ('rl-aux0','rl-aux2'):
        verified=verify(root,branch)
        config=json.loads((root/(branch+'.json')).read_text())
        by_id={r['id']:r for r in select_rows(Path(config['release']))}
        targets=torch.tensor([by_id[k]['targets'] for k in config['request_ids']],dtype=torch.long)
        mask=torch.ones_like(targets,dtype=torch.bool);weight=config['auxiliary_weight']
        path=root/branch/'behavior.pt';saved=torch.load(path,map_location='cpu',weights_only=True)
        logp=saved['candidate_logprobs'];exact=exact_additive_logit_gradient(logp,targets,mask,weight)
        baseline=expected_rule_reward(logp,targets,mask,weight)
        sizes=[]
        for samples in (8,32,128):
            errors=[];aux_norm=[];final_norm=[];grpo_aux_norm=[]
            for seed in range(repeats):
                actions,_=sample_spans(logp,samples,torch.Generator().manual_seed(10000+seed))
                reward=rule_sample_rewards(actions,targets,mask,weight)['total']
                gradient=sampled_logit_gradient(logp,actions,reward,baseline)
                errors.append(float((gradient-exact).square().mean()))
                aux_norm.append(float(gradient[:,:-1].norm(dim=(1,2)).mean()))
                final_norm.append(float(gradient[:,-1].norm(dim=1).mean()))
                adv=(reward-reward.mean(-1,keepdim=True))/(reward.std(-1,keepdim=True)+1e-6)
                grpo=sampled_logit_gradient(logp,actions,adv,torch.zeros(len(targets)))
                grpo_aux_norm.append(float(grpo[:,:-1].norm(dim=(1,2)).mean()))
            sizes.append(dict(samples=samples,repeats=repeats,score_gradient_mse=sum(errors)/repeats,
                mean_auxiliary_logit_gradient_norm=sum(aux_norm)/repeats,
                mean_final_logit_gradient_norm=sum(final_norm)/repeats,
                grpo_standardized_auxiliary_logit_gradient_norm=sum(grpo_aux_norm)/repeats))
        reports.append(dict(branch=branch,auxiliary_weight=weight,behavior_sha256=file_hash(path),
            initial_policy_id=saved['policy_id'],requests=len(targets),
            exact_auxiliary_logit_gradient_norm=float(exact[:,:-1].norm(dim=(1,2)).mean()),
            exact_final_logit_gradient_norm=float(exact[:,-1].norm(dim=1).mean()),estimates=sizes))
    return dict(passed=True,reports=reports,optimizer_updates=0,
        scope='CPU finite-policy gradient diagnostic from verified learning behavior. Logit-coordinate credit, not shared-parameter gradient or task gains. Exact expected-reward oracle differs from finite-sample standardized GRPO; no new training objective admitted.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=diagnose(a.root);atomic_json(a.output,r);print(json.dumps(r,indent=2))

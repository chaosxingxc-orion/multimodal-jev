#!/usr/bin/env python3
"""Offline pair-credit diagnostic from previously authenticated learning updates."""
import argparse
import json
from pathlib import Path
import torch

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_pair_credit import learning_pairs,exact_pair_credit,expected_pair_reward
from multimodal_jev.span_transfer_runtime import step_rows,branch_protocol


def diagnose(verified_updates):
    accepted=json.loads(verified_updates.read_text())
    audit=accepted['chain_audit']
    if accepted.get('passed') is not True or audit.get('passed') is not True:
        raise ValueError('verified update diagnostic required')
    root=Path(accepted['training_root'])
    if Path(audit['root'])!=root or len(accepted['updates'])!=audit['optimizer_updates']:
        raise ValueError('verified chain identity differs')
    for path,digest in accepted['files'].items():
        if file_hash(path)!=digest:raise ValueError('authenticated diagnostic input changed')
    result=[]
    for item in accepted['updates']:
        stage=root/'run'/f"update-{item['update']:04d}"
        config=json.loads((stage/'config.json').read_text());rows=step_rows(config)
        eligible=branch_protocol(config)['eligible_training_request_ids']
        pairs=learning_pairs(rows,eligible);targets=torch.tensor([r['targets'][-1] for r in rows],dtype=torch.long)
        saved=[json.loads((stage/'run'/(name+'.json')).read_text()) for name in ('before','after')]
        old,new=[torch.tensor(r['candidate_logprobs'][-1],dtype=torch.float64) for r in saved]
        p_gold=old.exp().gather(1,targets[:,None]).squeeze(1)
        scenarios=[]
        for weight in (0.,.5,1.):
            adv=exact_pair_credit(old,targets,pairs,weight)
            ascent=old.exp()*adv/len(pairs)
            # Hypothetical objective, evaluated on the SAME already executed update.
            before=expected_pair_reward(old,targets,pairs,weight).mean()
            after=expected_pair_reward(new,targets,pairs,weight).mean()
            paired_p=p_gold[pairs]
            partner_scale=(1-weight)+2*weight*paired_p.flip(-1)
            scenarios.append(dict(pair_weight=weight,expected_before=float(before),expected_after=float(after),
                actual_update_reward_delta=float(after-before),
                linearized_actual_update_reward_delta=float((ascent*(new-old)).sum()),
                conditional_gradient_norm=float(ascent.norm()),
                scale_relative_to_mean_correctness_min=float(partner_scale.min()),
                scale_relative_to_mean_correctness_max=float(partner_scale.max())))
        result.append(dict(update=item['update'],pairs=len(pairs),scenarios=scenarios))
    return dict(passed=True,training_admitted=False,optimizer_updates=0,
        verified_update_report=str(verified_updates),verified_update_report_sha256=file_hash(verified_updates),
        files=accepted['files'],updates=result,
        scope='Learning-only post-hoc hypothetical pair objectives on existing policy changes. '
              'No candidate objective was trained; no new inference or evaluation gold consumed. '
              'Candidate weights are diagnostic endpoints, not selected hyperparameters. '
              'Pure pair reward has a stationary 0.5 prediction solution under tied polarities.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verified-updates',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=diagnose(a.verified_updates);atomic_json(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','updates')},indent=2))
    for r in result['updates']:
        print(r['update'],[(x['pair_weight'],round(x['actual_update_reward_delta'],6)) for x in r['scenarios']])

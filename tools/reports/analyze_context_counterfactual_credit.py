"""Offline singleton-F1 credit diagnostic, never a runtime algorithm change.

Exact expectation differentiates the finite Bernoulli reward distribution.
Conditional direction integrates a node's own action, retaining the archived
other-node samples. Neither is an actor parameter-gradient or efficacy result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from tools.reports.replay_context_reward_groups import run as authenticate, identity


def exact_direction(probabilities, gold):
    p = torch.as_tensor(probabilities, dtype=torch.float64)
    if (p.ndim != 1 or not len(p) or not torch.isfinite(p).all()
            or ((p <= 0) | (p >= 1)).any() or type(gold) is not int or not 0 <= gold < len(p)):
        raise ValueError('complete interior Bernoulli singleton truth required')
    logits = torch.logit(p).requires_grad_()
    q = logits.sigmoid()
    mass = q.new_ones(1)
    for j in range(len(q)):
        if j == gold: continue
        mass = torch.nn.functional.pad(mass*(1-q[j]), (0,1)) + torch.nn.functional.pad(mass*q[j], (1,0))
    reward = q[gold] * (mass * (2/(torch.arange(len(mass), dtype=q.dtype)+2))).sum()
    gradient, = torch.autograd.grad(reward, logits)
    return float(reward.detach()), gradient.detach()


def conditional_direction(probabilities, gold, actions):
    p = torch.as_tensor(probabilities, dtype=torch.float64)
    if (actions.ndim != 2 or actions.shape[1] != len(p) or not len(actions)
            or not ((actions == 0) | (actions == 1)).all()):
        raise ValueError('complete binary actions required')
    # Validate probability/truth contract independently of sampled rewards.
    if (not torch.isfinite(p).all() or ((p <= 0) | (p >= 1)).any()
            or type(gold) is not int or not 0 <= gold < len(p)):
        raise ValueError('complete interior Bernoulli singleton truth required')
    a = actions.double(); counts = a.sum(-1); directions = []
    for j in range(len(p)):
        others = counts-a[:,j]
        if j == gold:
            difference = 2/(others+2)
        else:
            difference = 2*a[:,gold]*(1/(others+2)-1/(others+1))
        directions.append(p[j]*(1-p[j])*difference)
    return torch.stack(directions, dim=1)


def run(observation, release, schedule, receipts, cache_proof):
    bound = authenticate(observation, release, schedule, receipts, cache_proof)
    raw = json.loads(Path(observation).read_text())['progress']['latest'][0]['data']['rollout']['queries']
    rows = []
    for query, audited in zip(raw, bound['queries'], strict=True):
        p = torch.tensor(query['behavior_record']['prob'], dtype=torch.float32)
        sample = query['rollout']; gold = audited['gold_index']
        actions = (torch.rand((1,16,60), generator=torch.Generator().manual_seed(sample['seed'])) < p).long()
        if identity(actions) != audited['action_identity']: raise ValueError('archived actions differ')
        expected, exact = exact_direction(p, gold)
        per_sample = conditional_direction(p, gold, actions[0]); conditional = per_sample.mean(0)
        rewards = 2*actions[0,:,gold].double()/(actions[0].sum(-1)+1)
        centered = rewards-rewards.mean()
        rloo = ((16/15)*centered[:,None]*(actions[0].double()-p.double())).mean(0)
        norm = float(exact.norm())
        cosine = (float(torch.nn.functional.cosine_similarity(conditional[None], exact[None], eps=1e-30))
                  if norm and float(conditional.norm()) else None)
        rows.append(dict(request_id=audited['request_id'], source_group=audited['source_group'],
            reward_kind=audited['reward_kind'], threshold_exact_set=audited['threshold_exact_set'],
            expected_F1=expected, gold_probability=float(p[gold]),
            exact_direction=exact.tolist(), conditional_direction=conditional.tolist(),
            rloo_direction=rloo.tolist(), exact_l2=norm, conditional_l2=float(conditional.norm()),
            exact_gold=float(exact[gold]), conditional_gold=float(conditional[gold]),
            conditional_error_l2=float((conditional-exact).norm()),
            rloo_error_l2=float((rloo-exact).norm()), conditional_exact_cosine=cosine))
    strata = {}
    for name, subset in [('all',rows),('constant',[r for r in rows if r['reward_kind']!='varying']),
                         ('constant_wrong',[r for r in rows if r['reward_kind']!='varying' and not r['threshold_exact_set']])]:
        strata[name] = dict(queries=len(subset),
            nonzero_exact=sum(r['exact_l2']>0 for r in subset),
            nonzero_conditional=sum(r['conditional_l2']>0 for r in subset),
            exact_gold_min=min((r['exact_gold'] for r in subset),default=None),
            exact_gold_max=max((r['exact_gold'] for r in subset),default=None),
            mean_conditional_squared_error=sum(r['conditional_error_l2']**2 for r in subset)/max(1,len(subset)),
            mean_rloo_squared_error=sum(r['rloo_error_l2']**2 for r in subset)/max(1,len(subset)))
    sources=dict(bound['sources']);sources[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return dict(schema='context-singleton-counterfactual-credit-diagnostic-v1', sources=sources,
        actor_update=bound['actor_update'], behavior_update=bound['behavior_update'], strata=strata,queries=rows,
        limits=['One existing TRAIN batch; no independent sampling trials or heldout results.',
                'Directions are reward-ascent derivatives with respect to output logits, not model parameter gradients.',
                'Conditional estimator integrates its own Bernoulli bit; this is stronger than sampled node-baseline subtraction.',
                'Unbiasedness concerns unnormalized expected reward at the behavior policy, not GRPO normalization or clipped off-policy loss.',
                'Same samples give descriptive estimation errors, not a general variance or training improvement claim.',
                'p*(1-p) remains: nonzero signal need not be large enough to correct saturated errors.',
                'Singleton truth only; no new process gold, no missing-label imputation, no runtime change.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('observation','release','schedule','receipts','cache-proof','output'):parser.add_argument('--'+key,required=True)
    a=parser.parse_args();result=run(a.observation,a.release,a.schedule,a.receipts,a.cache_proof)
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result['strata'],indent=2))

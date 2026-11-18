"""Probability-space diagnosis of committed TRAIN records, not actor replay."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from tools.reports.replay_context_reward_groups import run as authenticate, identity
from tools.jev.context_online_batches import auxiliary_coefficient


def components(probability, reference, gold):
    p = torch.as_tensor(probability, dtype=torch.float64)
    q = torch.as_tensor(reference, dtype=torch.float64)
    if (p.ndim != 1 or p.shape != q.shape or len(p) < 2
            or not torch.isfinite(p).all() or not torch.isfinite(q).all()
            or ((p <= 0) | (p >= 1) | (q <= 0) | (q >= 1)).any()
            or type(gold) is not int or not 0 <= gold < len(p)):
        raise ValueError('complete interior candidate probabilities required')
    z = torch.logit(p).requires_grad_()
    current = z.sigmoid()
    mask = torch.arange(len(p)) != gold
    positive = -current[gold].log()
    negative = -torch.log1p(-current[mask]).sum()
    nll = (positive+negative)/len(p)
    rank = F.softplus(z[mask]-z[gold]).mean()
    kl = (current*(current.log()-q.log())+(1-current)*(torch.log1p(-current)-torch.log1p(-q))).sum()
    losses = {'ranking': rank, 'binary_nll': nll, 'reference_kl_sum': kl}
    gradients = {k: torch.autograd.grad(v, z, retain_graph=True)[0].detach() for k, v in losses.items()}
    scores = {k: float(v.detach()) for k, v in losses.items()}
    scores.update(positive_nll=float(positive.detach()), negative_nll_sum=float(negative.detach()),
                  positive_contribution=float(positive.detach())/len(p),
                  negative_contribution=float(negative.detach())/len(p),
                  gold_probability=float(p[gold]),
                  confident_false_negative=int(p[gold] < .01),
                  confident_false_positives=int((p[mask] > .99).sum()))
    return scores, gradients


def run(observation, release, schedule, receipts, cache_proof):
    bound = authenticate(observation, release, schedule, receipts, cache_proof)
    queries = json.loads(Path(observation).read_text())['progress']['latest'][0]['data']['rollout']['queries']
    alpha = auxiliary_coefficient(bound['behavior_update'])
    rows = []
    for query, truth in zip(queries, bound['queries'], strict=True):
        p = torch.tensor(query['behavior_record']['prob'], dtype=torch.float32)
        reference = query['reference_record']['prob']; gold = truth['gold_index']
        values, gradients = components(p.double(), reference, gold)
        ref_values, _ = components(reference, reference, gold)
        a = (torch.rand((1,16,len(p)), generator=torch.Generator().manual_seed(query['rollout']['seed'])) < p).long()
        if identity(a) != truth['action_identity']: raise ValueError('action identity mismatch')
        rewards = 2*a[0,:,gold].float()/(a[0].sum(-1)+1)
        advantage = (rewards-rewards.mean())/(rewards.std(correction=1)+1e-6)
        # Negative reward-ascent direction: minimizes surrogate at ratio=1.
        grads = dict(grpo_at_ratio_one=-(advantage[:,None].double()*(a[0].double()-p.double())).mean(0),
            ranking=alpha*gradients['ranking'], binary_nll=alpha*gradients['binary_nll'],
            reference_kl=.01*gradients['reference_kl_sum'])
        combined = sum(grads.values())
        rows.append(dict(request_id=truth['request_id'], gold_index=gold,
            threshold_exact_set=truth['threshold_exact_set'], behavior=values, reference=ref_values,
            loss_gradient_l2={k: float(v.norm()) for k,v in grads.items()},
            loss_gradient_gold={k: float(v[gold]) for k,v in grads.items()},
            combined_logit_l2=float(combined.norm()),
            gold_descent_increases_probability=bool(combined[gold]<0),
            nll_ascent_inner_product=float(torch.dot(gradients['binary_nll'], -combined)),
            negative_nll_delta=values['negative_contribution']-ref_values['negative_contribution'],
            positive_nll_delta=values['positive_contribution']-ref_values['positive_contribution']))
    summary={}
    for label, subset in [('all',rows),('correct',[r for r in rows if r['threshold_exact_set']]),
                          ('wrong',[r for r in rows if not r['threshold_exact_set']])]:
        summary[label]=dict(queries=len(subset),
            scores={s:{k:sum(r[s][k] for r in subset)/len(subset) for k in subset[0][s]} for s in ('behavior','reference')},
            mean_component_logit_gradient_l2={k:sum(r['loss_gradient_l2'][k] for r in subset)/len(subset) for k in subset[0]['loss_gradient_l2']},
            mean_positive_nll_delta=sum(r['positive_nll_delta'] for r in subset)/len(subset),
            mean_negative_nll_delta=sum(r['negative_nll_delta'] for r in subset)/len(subset),
            local_euclidean_direction_increases_nll=sum(r['nll_ascent_inner_product']>1e-12 for r in subset),
            gold_direction_increases_probability=sum(r['gold_descent_increases_probability'] for r in subset))
    sources=dict(bound['sources'])
    for path in (Path(__file__),Path('tools/jev/context_online_batches.py')):
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(schema='context-probability-objective-diagnostic-v1',actor_update=bound['actor_update'],
        behavior_update=bound['behavior_update'],alpha=alpha,kl_coefficient=.01,sources=sources,
        summary=summary,queries=rows,limits=[
            'Effective logits reconstructed from stored FP32 probabilities, not original BF16 actor logits.',
            'Output-space local derivatives only; no shared-network Jacobian, Adam/FSDP or post-update probabilities.',
            'GRPO uses independently replayed formula at ratio1; no native actor loss reproduction.',
            'Gradient norms are per-query before common batch division; not parameter update magnitudes.',
            'Gold inverse-sigmoid ranking is approximate relative to original logits; NLL/KL evaluated from probabilities.',
            'TRAIN snapshot only, no selection efficacy or hyperparameter selection.'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for key in ('observation','release','schedule','receipts','cache-proof','output'):parser.add_argument('--'+key,required=True)
    args=vars(parser.parse_args());output=args.pop('output');torch.set_num_threads(1)
    result=run(**args)
    with Path(output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'alpha':result['alpha'],'summary':result['summary']},indent=2))

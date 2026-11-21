"""Editable simple-task GRPO objective, probability-margin candidate v3; requires NPU admission.

Use authenticated node samples as complete independent Bernoulli set samples.
Task rewards use JOINT set log probabilities, never a broadcast node ratio.
Coefficients operate after advantage normalization. No extra model forward.
"""
import math
import torch
from torch.nn import functional as F

from multimodal_jev import context_dual_credit as dual
from multimodal_jev.context_task_reward import score
from multimodal_jev.objective_plugin import continuous_advantages

API_VERSION = 'jev-stage2-objective-v1'
DEFAULTS = dict(ranking_rl_weight=.25, existence_rl_weight=.25, task_rl_weight=.5,
    exact_weight=.5, overlap_weight=.5, beta=1., ranking_auxiliary_coefficient=.1,
    probability_nll_sum_coefficient=.05, reference_kl_mean_coefficient=.01, probability_margin=.1, probability_margin_coefficient=.1)


def options(ctx):
    if getattr(ctx,'arm','joint') != 'joint':
        raise ValueError('this task objective requires the joint arm')
    if set(ctx.options) != set(DEFAULTS):
        raise ValueError('explicit complete task-objective options required')
    c=dict(ctx.options)
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in c.values()):
        raise ValueError('finite nonnegative objective coefficients required')
    if c["probability_margin"]>=1:raise ValueError("probability margin must be less than one")
    return c


def compute_rewards(ctx):
    c=options(ctx); result=[]
    for record in ctx.records:
        branches=record['branches']; existence=branches['existence']
        actions=existence['actions'].T[None,:,:]
        labels=existence['target'][None,:]
        task=score(actions,labels,torch.ones_like(labels),dict(schema='jev-task-set-reward-v1',
            semantics='exclusive_event',exact_weight=c['exact_weight'],
            overlap_weight=c['overlap_weight'],beta=c['beta']))
        result.append(dict(ranking=branches['ranking']['rewards'].clone(),
            existence=existence['rewards'].clone(),task=task['reward'],
            task_exact=task['exact'],task_partial=task['partial']))
    return result


def compute_loss(ctx, rewards):
    c=options(ctx); z=ctx.logits.float(); real=ctx.real_requests.to(z.device)
    zero=z.sum()*0; rl=zero; rank_rl=zero; exist_rl=zero; task_rl=zero
    zero_groups=0; groups=0; task_exact=0.; task_partial=0.; samples=0
    for i,(record,identity,reward) in enumerate(zip(ctx.records,ctx.identities,rewards,strict=True)):
        terms=dual.current_terms(ctx.logits[i],record,**identity)
        active=int(real[i]); algorithm=record['binding']['algorithm']
        for name,weight in [('ranking',c['ranking_rl_weight']),('existence',c['existence_rl_weight'])]:
            terms[name]['advantages']=continuous_advantages(reward[name],algorithm)
            contribution=dual.native_surrogate(terms[name]).to(z.device)*active
            if name=='ranking':rank_rl=rank_rl+weight*contribution
            else:exist_rl=exist_rl+weight*contribution
        # Each column of the independently seeded candidate samples is one
        # authentic whole-set action under the behavior Bernoulli product.
        node=terms['existence']
        joint=dict(new_logprob=node['new_logprob'].double().sum(0,keepdim=True),
            old_logprob=node['old_logprob'].double().sum(0,keepdim=True),
            advantages=continuous_advantages(reward['task'],algorithm),
            weight=torch.ones(1,dtype=torch.float32))
        task_rl=task_rl+c['task_rl_weight']*dual.native_surrogate(joint).to(z.device)*active
        zero_groups+=int(reward['task'].std(unbiased=False)==0)*active;groups+=active
        task_exact+=float(reward['task_exact'].sum())*active
        task_partial+=float(reward['task_partial'].sum())*active
        samples+=reward['task'].numel()*active
    scale=ctx.world_size/ctx.global_real_requests
    rank_rl=rank_rl*scale;exist_rl=exist_rl*scale;task_rl=task_rl*scale
    rl=rank_rl+exist_rl+task_rl
    labels=torch.tensor([r['binding']['labels'] for r in ctx.records],device=z.device,dtype=torch.bool)
    if not (labels.sum(-1)==1).all(): raise ValueError('complete exclusive-event truth required')
    # Uniform Bernoulli log score preserves marginal-probability semantics.
    # A sum (explicitly named) keeps a saturated wrong bit from being divided
    # by all 60 candidates. No class reweighting or target-prior distortion.
    nll=F.binary_cross_entropy_with_logits(z,labels.float(),reduction='none').sum(-1)
    gold=z.masked_select(labels).reshape(-1,1)
    negatives=z.masked_select(~labels).reshape(z.shape[0],-1)
    # Emphasize the largest distractor; this is auxiliary ranking supervision.
    rank=F.softplus(negatives.max(-1).values-gold[:,0])
    nll_total=(nll*real).sum()*scale;rank_total=(rank*real).sum()*scale
    auxiliary=c['probability_nll_sum_coefficient']*nll_total+c['ranking_auxiliary_coefficient']*rank_total
    probabilities=z.sigmoid()
    margin_gap=probabilities.masked_select(labels).reshape(-1)-probabilities.masked_fill(labels,float('-inf')).max(-1).values
    margin_loss=(F.relu(c['probability_margin']-margin_gap)*real).sum()*scale
    auxiliary=auxiliary+c['probability_margin_coefficient']*margin_loss
    p=probabilities.to('cpu');logs=dual.log_probabilities(p)
    reference=dual.log_probabilities(ctx.reference_probability)
    per_query=(torch.stack((1-p,p),-1)*(logs-reference)).sum(-1).mean(-1)
    kl=(per_query*ctx.real_requests.cpu()).sum()*scale
    total=rl+auxiliary+c['reference_kl_mean_coefficient']*kl.to(z.device)
    return total,dict(rl=float(rl.detach()),auxiliary=float(auxiliary.detach()),kl=float(kl.detach()),
        ranking_rl=float(rank_rl.detach()),existence_rl=float(exist_rl.detach()),task_rl=float(task_rl.detach()),
        nll_sum=float(nll_total.detach()),hardest_pair=float(rank_total.detach()),
        probability_margin_loss=float(margin_loss.detach()),
        probability_margin_gap_sum=float((margin_gap*real).sum().detach()),
        probability_margin_satisfied_count=int(((margin_gap>=c["probability_margin"])*real).sum()),
        task_zero_variance_groups=zero_groups,real_groups=groups,
        sampled_exact_count=task_exact,sampled_partial_count=task_partial,sampled_set_count=samples,
        scope='Dual credit plus sampled joint-set GRPO; summed NLL and mean reference KL. Candidate, not an admitted NPU result.')

"""Exact additive credit for ragged categorical Span decisions.

Marginal PPO with old-policy quadrature, not joint-action PPO or set-F1
optimization. Rewards/masks come from authenticated supervision outside actor
inputs. Caller owns immutable policy identity and full backend parity gates.
"""
import math
import torch


def validate_policy(policy, batch, detached=False):
    if not policy:raise ValueError('nonempty categorical policy required')
    device=policy[0].device
    for p in policy:
        if (p.ndim!=2 or p.shape[0]!=batch or p.shape[1]<2
                or p.dtype not in (torch.float32,torch.float64) or p.device!=device
                or (detached and p.requires_grad) or not torch.isfinite(p).all()
                or not torch.allclose(p.logsumexp(-1),p.new_zeros(batch),atol=1e-6,rtol=0)):
            raise ValueError('aligned normalized FP32/FP64 categorical policy required')


def credit(behavior,targets,supervised,span_weights,request_weights=None):
    """A_j(a)=w_j*(1[a=y_j]-p0_j(y_j)); missing labels receive zero credit.

    span_weights fix each request's task budget; request_weights control the
    cross-request mean. They are not standardized within a sampled group.
    Requests without any positively weighted reviewed decision are excluded
    from the denominator, never included as zero-reward negative examples.
    """
    if (targets.ndim!=2 or targets.dtype!=torch.long or not len(targets)
            or targets.shape[1]!=len(behavior) or supervised.shape!=targets.shape
            or supervised.dtype!=torch.bool or span_weights.shape!=targets.shape
            or span_weights.requires_grad or not span_weights.is_floating_point()
            or not torch.isfinite(span_weights).all() or (span_weights<0).any()):
        raise ValueError('aligned detached nonnegative credit weights and supervision required')
    validate_policy(behavior,len(targets),detached=True)
    device=behavior[0].device
    if any(t.device!=device for t in (targets,supervised,span_weights)):
        raise ValueError('credit tensors must share policy device')
    weights=span_weights*supervised
    if request_weights is None:request_weights=weights.new_ones(len(targets))
    if (request_weights.shape!=(len(targets),) or request_weights.requires_grad
            or request_weights.device!=device or not request_weights.is_floating_point()
            or not torch.isfinite(request_weights).all() or (request_weights<0).any()):
        raise ValueError('finite detached nonnegative request weights required')
    request_weights=request_weights*(weights.sum(-1)>0)
    if request_weights.sum()<=0:raise ValueError('at least one weighted reviewed decision required')
    advantages=[];probabilities=[];gold_probabilities=[]
    for j,p in enumerate(behavior):
        known=supervised[:,j];y=targets[:,j]
        if ((y[known]<0)|(y[known]>=p.shape[1])).any():raise ValueError('reviewed target outside candidate support')
        safe=y.masked_fill(~known,0)
        onehot=torch.nn.functional.one_hot(safe,p.shape[1]).to(p)
        probability=p.exp();gold_probability=(probability*onehot).sum(-1)
        advantages.append(((onehot-gold_probability[:,None])*weights[:,j:j+1]).detach())
        probabilities.append(probability.detach())
        gold_probabilities.append(gold_probability.masked_fill(~known,float('nan')))
    return dict(advantages=tuple(advantages),probabilities=tuple(probabilities),
                span_weights=weights.detach(),request_weights=request_weights.detach(),
                gold_probabilities=torch.stack(gold_probabilities,-1).detach())


def loss_inputs(behavior,actor,targets,supervised,span_weights,request_weights=None,clip_ratio=.2):
    if type(clip_ratio) not in (int,float) or not math.isfinite(clip_ratio) or not 0<clip_ratio<1:
        raise ValueError('bounded clip ratio required')
    result=credit(behavior,targets,supervised,span_weights,request_weights)
    validate_policy(actor,len(targets))
    if len(actor)!=len(behavior) or any(a.shape!=b.shape or a.device!=b.device for a,b in zip(actor,behavior,strict=True)):
        raise ValueError('actor candidate support differs')
    # Native token-mean divides by every flattened candidate. Compensate this
    # exactly to preserve a weighted request mean regardless of support sizes.
    total=sum(p.numel() for p in behavior)
    rw=result['request_weights'];normalizer=rw.sum()
    quadrature=tuple(p*rw[:,None]*total/normalizer for p in result['probabilities'])
    flatten=lambda xs:torch.cat([x.reshape(-1,1) for x in xs],0)
    return tuple(flatten(xs) for xs in (behavior,actor,result['advantages'],quadrature))


def reference_loss(behavior,actor,targets,supervised,span_weights,request_weights=None,clip_ratio=.2):
    old,new,adv,weight=loss_inputs(behavior,actor,targets,supervised,span_weights,request_weights,clip_ratio)
    ratio=(new-old).clamp(-20,20).exp()
    clipped=torch.maximum(-adv*ratio,-adv*ratio.clamp(1-clip_ratio,1+clip_ratio))
    dual=torch.where(adv<0,torch.minimum(-adv*3.,clipped),clipped)
    return (dual*weight).mean()


def verl_loss(behavior,actor,targets,supervised,span_weights,request_weights=None,clip_ratio=.2):
    from omegaconf import OmegaConf
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
    old,new,adv,weight=loss_inputs(behavior,actor,targets,supervised,span_weights,request_weights,clip_ratio)
    config=OmegaConf.create(dict(clip_ratio=clip_ratio,clip_ratio_low=None,clip_ratio_high=None,
                                clip_ratio_c=3.,global_batch_info={}))
    return compute_policy_loss_vanilla(old,new,adv,torch.ones_like(adv),config=config,rollout_is_weights=weight)


def correctness_sampling_risk(behavior,targets,supervised,span_weights,samples):
    """Per-Span Bernoulli correctness risks, not multi-Span total-reward risk."""
    if type(samples) is not int or samples<2:raise ValueError('at least two samples required')
    info=credit(behavior,targets,supervised,span_weights)
    p=info['gold_probabilities']
    return dict(gold_probability=p,all_wrong_probability=(1-p).pow(samples),
                constant_correctness_probability=(1-p).pow(samples)+p.pow(samples),
                scope='Per individual Span correctness; not joint additive-reward variance.')

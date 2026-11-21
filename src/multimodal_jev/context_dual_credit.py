"""Dual binary credit contracts; no model, engine, or trainer dispatch.

Rank probabilities use CPU FP64 differences of actual head logits on both
sides. Existence uses the actual FP32 sigmoid path of the caller's backend.
Only the small decision tensors cross CPU; current-policy autograd is kept.
The native surrogate adapter imports and pins verl only when called.
"""
from dataclasses import dataclass
import hashlib
import inspect
import json

import torch

from .span_group_advantage import CORE_SHA256
from .baseline import file_hash


ALGORITHMS = ('dual_credit_rloo_clipped_v1', 'dual_credit_grpo_floor_clipped_v1')
ARMS = ('ranking_only', 'existence_only', 'joint')


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _interior(p, *, dtype=torch.float32):
    if p.dtype != dtype or not torch.isfinite(p).all() or ((p <= 0) | (p >= 1)).any():
        raise ValueError('actual finite interior probabilities in declared dtype required; no clipping')


def _truth(candidate_ids, labels):
    if (not candidate_ids or any(type(x) is not str or not x for x in candidate_ids)
            or len(set(candidate_ids)) != len(candidate_ids)
            or labels.dtype != torch.bool or labels.shape != (len(candidate_ids),)):
        raise ValueError('distinct candidate identities and complete Boolean labels required')


@dataclass
class Components:
    keys: tuple
    probability: torch.Tensor  # M, CPU FP64 ranking / FP32 existence, possibly differentiable
    target: torch.Tensor       # M, CPU Boolean
    weight: torch.Tensor       # M, CPU FP32; sum=1 unless M=0


def components(logits, candidate_ids, labels, *, actual_probability=None):
    """One request; candidate order is canonicalized by IDs, never by labels.

    actual_probability is for the detached engine record only. The actor must
    call without it, recomputing differentiable p from its actual forward.
    Missing supervision is not represented by this complete-truth contract.
    """
    _truth(candidate_ids, labels)
    if (logits.ndim != 1 or logits.shape != labels.shape
            or logits.dtype not in (torch.float32, torch.bfloat16)
            or not torch.isfinite(logits).all()):
        raise ValueError('finite BF16 or FP32 head logits for complete candidates required')
    if actual_probability is not None:
        if logits.requires_grad or actual_probability.requires_grad or actual_probability.shape != logits.shape:
            raise ValueError('external probabilities only allowed on detached behavior records')
        p = actual_probability.to('cpu')
    else:
        p = logits.float().sigmoid().to('cpu')
    _interior(p)
    z = logits.float().to('cpu'); y = labels.to('cpu')
    indices = sorted(range(len(candidate_ids)), key=lambda i: candidate_ids[i])
    idx = torch.tensor(indices, dtype=torch.long)
    cp, cy = p[idx], y[idx]
    positives = int(cy.sum()); negatives = len(cy) - positives
    weights = torch.empty_like(cp)
    if positives and negatives:
        weights[cy] = .5 / positives; weights[~cy] = .5 / negatives
    else:
        weights.fill_(1. / len(cy))
    pairs = [(i,j) for at,i in enumerate(indices) for j in indices[at+1:] if bool(y[i] != y[j])]
    if pairs:
        left = torch.tensor([i for i,j in pairs]); right = torch.tensor([j for i,j in pairs])
        qp = (z[left].double()-z[right].double()).sigmoid()  # same CPU FP64 path for actor and sampler
        _interior(qp,dtype=torch.float64)
        qt = y[left]; qw = torch.full((len(pairs),),1./len(pairs),dtype=torch.float32)
    else:
        qp = z[:0].double(); qt = y[:0]; qw = p[:0].detach()
    return {
        'ranking': Components(tuple((candidate_ids[i],candidate_ids[j]) for i,j in pairs), qp, qt, qw),
        'existence': Components(tuple((candidate_ids[i],) for i in indices), cp, cy, weights)
    }


def log_probabilities(p):
    if p.dtype not in (torch.float32,torch.float64):raise ValueError('FP32 existence or FP64 ranking required')
    _interior(p,dtype=p.dtype)
    return torch.stack((torch.log1p(-p), torch.log(p)), dim=-1)


def advantages(rewards, algorithm):
    if (algorithm not in ALGORITHMS or rewards.ndim != 2 or rewards.shape[1] < 2
            or rewards.requires_grad or rewards.dtype != torch.float32
            or not torch.isfinite(rewards).all() or not ((rewards == 0) | (rewards == 1)).all()):
        raise ValueError('detached binary M x G rewards and registered algorithm required')
    if algorithm == ALGORITHMS[0]:
        return rewards - (rewards.sum(-1, keepdim=True)-rewards)/(rewards.shape[1]-1)
    centered = rewards - rewards.mean(-1, keepdim=True)
    return centered / rewards.std(-1, unbiased=False, keepdim=True).clamp_min(.25)


def _binding(logits, probability, candidate_ids, labels, *, policy_id, input_id,
             query_id, update, algorithm, group_size, seed):
    if any(type(v) is not str or not v for v in (policy_id,input_id,query_id)):
        raise ValueError('explicit policy/input/schedule query identity required')
    if any(type(v) is not int or v < 0 for v in (update,seed)) or type(group_size) is not int or group_size < 2:
        raise ValueError('explicit update, seed and complete group required')
    if algorithm not in ALGORITHMS:
        raise ValueError('unregistered algorithm')
    return dict(policy_id=policy_id,input_id=input_id,query_id=query_id,update=update,
        algorithm=algorithm,group_size=group_size,seed=seed,candidates=list(candidate_ids),
        labels=labels.cpu().tolist(),logits=logits.float().cpu().tolist(),
        probability=probability.cpu().tolist(),arithmetic='dual-credit-cpu-pair-fp64-v2')


def prepare(logits, probability, candidate_ids, labels, *, policy_id, input_id,
            query_id, update, algorithm=ALGORITHMS[0], group_size=16, seed=317):
    """Sample authenticatable groups; no labels are supplied to the engine.

    Caller must independently authenticate logits/probability to its real
    engine reply. A self-consistent synthetic record is not execution proof.
    """
    if logits.requires_grad or probability.requires_grad:
        raise ValueError('detached behavior required')
    panel = components(logits,candidate_ids,labels,actual_probability=probability)
    binding = _binding(logits,probability,candidate_ids,labels,policy_id=policy_id,input_id=input_id,
        query_id=query_id,update=update,algorithm=algorithm,group_size=group_size,seed=seed)
    branches = {}
    for name, c in panel.items():
        samples = []
        for key, p in zip(c.keys,c.probability,strict=True):
            local_seed = int(_hash([seed,update,query_id,name,key])[:16],16) % (2**63)
            generator = torch.Generator(device='cpu').manual_seed(local_seed)
            samples.append((torch.rand(group_size,generator=generator,dtype=c.probability.dtype) < p).long())
        actions = torch.stack(samples) if samples else torch.empty((0,group_size),dtype=torch.long)
        rewards = (actions == c.target[:,None]).float()
        logs = log_probabilities(c.probability)
        old = logs.gather(1,actions) if samples else torch.empty((0,group_size),dtype=c.probability.dtype)
        branches[name] = dict(keys=c.keys,probability=c.probability,weight=c.weight,
            target=c.target,actions=actions,rewards=rewards,old_logprob=old,
            advantages=advantages(rewards,algorithm))
    return dict(binding=binding,binding_sha256=_hash(binding),branches=branches)


def verify(record):
    """Recompute every sampled field, including labels/weights and seed scope."""
    b = record['binding']
    if record['binding_sha256'] != _hash(b):
        raise ValueError('binding changed')
    rebuilt = prepare(torch.tensor(b['logits'],dtype=torch.float32),
        torch.tensor(b['probability'],dtype=torch.float32),b['candidates'],
        torch.tensor(b['labels'],dtype=torch.bool),
        **{k:b[k] for k in ('policy_id','input_id','query_id','update','algorithm','group_size','seed')})
    if rebuilt['binding'] != b or set(record['branches']) != set(rebuilt['branches']):
        raise ValueError('branch contract changed')
    for name, expected in rebuilt['branches'].items():
        actual = record['branches'][name]
        if set(actual) != set(expected):
            raise ValueError('branch fields changed')
        for k,v in expected.items():
            a = actual[k]
            if isinstance(v, torch.Tensor):
                if (not isinstance(a,torch.Tensor) or a.requires_grad or a.device.type != 'cpu'
                        or a.dtype != v.dtype or a.shape != v.shape or not torch.equal(a,v)):
                    raise ValueError('sample/reward/advantage identity changed: '+name+'/'+k)
            elif a != v:
                raise ValueError('component identity changed')
    return rebuilt


def current_terms(logits, record, *, policy_id, input_id, query_id, update):
    """At the first/only update require the expected behavior and close numbers.

    Binds caller identities but does not prove loaded weights. Full engine and
    actor provenance remains mandatory at runtime. No stale PPO epochs here.
    """
    record = verify(record); b = record['binding']
    if any(b[k] != v for k,v in dict(policy_id=policy_id,input_id=input_id,
                                    query_id=query_id,update=update).items()):
        raise ValueError('stale behavior/input/update')
    panel = components(logits,b['candidates'],torch.tensor(b['labels'],dtype=torch.bool))
    terms = {}
    for name,c in panel.items():
        old = record['branches'][name]
        logs = log_probabilities(c.probability)
        if c.probability.numel():
            gap = (c.probability.detach()-old['probability']).abs().max().item()
            log_gap = (logs.detach()-log_probabilities(old['probability'])).abs().max().item()
            flips = ((c.probability.detach() >= .5) != (old['probability'] >= .5)).sum().item()
            if gap > 2e-7 or log_gap > 2e-6 or flips:
                raise ValueError('updated or numerically inconsistent behavior: '+name)
        terms[name] = dict(new_logprob=logs.gather(1,old['actions']),
            old_logprob=old['old_logprob'],advantages=old['advantages'],weight=old['weight'])
    return terms


from .grpo_numerics import current as grpo_settings


def native_surrogate(terms):
    """Per-query weighted component mean through the pinned verl surrogate."""
    from omegaconf import OmegaConf
    from verl.trainer.ppo import core_algos
    if file_hash(inspect.getfile(core_algos)) != CORE_SHA256:
        raise ValueError('pinned native surrogate changed')
    # Only the tiny rank distribution uses FP64; native verl loss and NPU
    # gradients remain FP32, with the same cast on current and behavior logs.
    new,old,adv = (terms[k].float() for k in ('new_logprob','old_logprob','advantages'))
    if new.shape[0] == 0:
        return new.sum()*0
    weights = terms['weight'][:,None].expand_as(new)
    active=grpo_settings()
    clip=(active['policy_clip'] if active is not None else
          dict(clip_ratio=.2,clip_ratio_low=None,clip_ratio_high=None,clip_ratio_c=3.))
    config=OmegaConf.create(dict(**clip,global_batch_info={}))
    value,_ = core_algos.compute_policy_loss_vanilla(old.reshape(-1,1),new.reshape(-1,1),
        adv.reshape(-1,1),torch.ones_like(new).reshape(-1,1),config=config,
        rollout_is_weights=weights.reshape(-1,1))
    return value * new.shape[0]  # native mean over M*G -> sum_M(weight*mean_G)


def rl_loss(logits, records, identities, real_requests, *, global_real_requests,
            world_size=1, arm='joint'):
    """RL part only; the unified objective adds supervision and reference once."""
    if (logits.ndim != 2 or len(records) != logits.shape[0] or len(identities) != len(records)
            or real_requests.dtype != torch.bool or real_requests.shape != (len(records),)
            or type(world_size) is not int or world_size < 1 or arm not in ARMS
            or type(global_real_requests) is not int or global_real_requests < max(1,int(real_requests.sum()))):
        raise ValueError('explicit global real-query denominator and valid batch required')
    total = logits.float().sum()*0
    for i,(record,identity) in enumerate(zip(records,identities,strict=True)):
        terms = current_terms(logits[i],record,**identity)
        for name in ('ranking','existence'):
            if arm not in ('joint',name+'_only'):
                continue
            contribution = native_surrogate(terms[name]).to(logits.device)
            total = total + .5*contribution*int(real_requests[i])
    return total * world_size/global_real_requests


def hybrid_loss(logits, records, identities, reference_probability, real_requests, *,
                global_real_requests, world_size=1, arm='joint'):
    """Single shared objective: dual RL + ranking .1 + NLL .3 + KL .01.

    Reference identity must be authenticated by the existing cache owner.
    This helper validates numeric distribution/shape, not remote provenance.
    """
    from .context_candidate_objective import loss as auxiliary_loss
    if (reference_probability.shape != logits.shape or reference_probability.requires_grad
            or reference_probability.device.type != 'cpu'):
        raise ValueError('detached CPU reference probabilities matching complete panel required')
    _interior(reference_probability)
    rl = rl_loss(logits,records,identities,real_requests,
        global_real_requests=global_real_requests,world_size=world_size,arm=arm)
    labels = torch.tensor([r['binding']['labels'] for r in records],
                          dtype=torch.bool,device=logits.device)
    real = real_requests.to(logits.device)
    weights = real.float()
    auxiliary, _ = auxiliary_loss(logits,labels,torch.ones_like(labels),weights,real,
        global_real_requests=global_real_requests,world_size=world_size,
        ranking_coefficient=.1,probability_coefficient=.3)
    p = logits.float().sigmoid().to('cpu'); current = log_probabilities(p)
    reference = log_probabilities(reference_probability)
    distribution = torch.stack((1-p,p),dim=-1)
    per_query = (distribution*(current-reference)).sum(dim=(-1,-2))
    kl = (per_query*real_requests.cpu()).sum()*world_size/global_real_requests
    total = rl + auxiliary + .01*kl.to(logits.device)
    return total,dict(rl=float(rl.detach()),auxiliary=float(auxiliary.detach()),
                      kl=float(kl.detach()),arm=arm,
                      scope='Loss interface only; engine/actor runtime not admitted')

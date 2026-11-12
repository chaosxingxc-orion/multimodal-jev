"""Experimental learning-only credit for BOTH complementary answers correct.

Independent requests remain independent model calls. This groups their rewards
offline; it introduces no inference loop or answer feeding. Not admitted by any
existing training protocol. Pure answer disagreement is deliberately not rewarded.
"""
import math
import torch


def learning_pairs(rows, eligible_ids):
    """Bind reviewed positive/negative rules by source identity, never row order."""
    groups = {}
    if (not rows or len({r['id'] for r in rows}) != len(rows)
            or not {r['id'] for r in rows} <= set(eligible_ids)):
        raise ValueError('unique learning rows required')
    for i,row in enumerate(rows):
        p = row['provenance']
        if (row['split'] != 'learning' or row.get('optimization_allowed',True) is not True
                or not row['targets'] or len(row['targets']) != len(row['supervised'])
                or any(v is not True for v in row['supervised'])
                or any(type(v) is not bool for v in row['targets'])):
            raise ValueError('fully reviewed optimization-eligible learning rows required')
        key = (p['source_group'],p['recording_id'],p['audio_sha256'],p['complement_pair_id'],row['arm'])
        group = groups.setdefault(key,{})
        polarity = p['rule_polarity']
        if polarity not in ('positive','negative') or polarity in group:
            raise ValueError('unique complementary polarities required')
        group[polarity] = i
    pairs = []
    for group in groups.values():
        if set(group) != {'positive','negative'}:
            raise ValueError('complete complementary pairs required')
        a,b = group['positive'],group['negative']
        left,right = rows[a],rows[b]
        if (left['targets'][:-1] != right['targets'][:-1]
                or left['targets'][-1] == right['targets'][-1]
                or right['provenance']['rule'] != {'op':'not','arg':left['provenance']['rule']}
                or any(left['provenance'][k] != right['provenance'][k]
                       for k in ('dependency_keys','dependency_targets'))):
            raise ValueError('reviewed complementary semantics differ')
        pairs.append([a,b])
    return torch.tensor(pairs,dtype=torch.long)


def validate(logprobs, targets, pairs, pair_weight):
    if (logprobs.ndim != 2 or logprobs.shape[1] != 2 or not torch.isfinite(logprobs).all()
            or not torch.allclose(logprobs.logsumexp(-1),logprobs.new_zeros(len(logprobs)),atol=1e-6,rtol=0)
            or targets.shape != (len(logprobs),) or targets.dtype != torch.long
            or ((targets < 0) | (targets > 1)).any()
            or pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype != torch.long
            or not len(pairs) or len(logprobs) != 2*len(pairs)
            or sorted(pairs.flatten().tolist()) != list(range(len(logprobs)))
            or type(pair_weight) not in (float,int) or not math.isfinite(pair_weight)
            or not 0 <= pair_weight <= 1):
        raise ValueError('normalized Boolean policies, exhaustive disjoint pairs and bounded weight required')
    if (targets[pairs[:,0]] == targets[pairs[:,1]]).any():
        raise ValueError('complementary reviewed gold required')


def expected_pair_reward(logprobs, targets, pairs, pair_weight):
    """Per-pair expected (1-beta)*mean(correct) + beta*both_correct."""
    validate(logprobs,targets,pairs,pair_weight)
    correct = logprobs.exp().gather(1,targets[:,None]).squeeze(1)[pairs]
    return (1-pair_weight)*correct.mean(-1)+pair_weight*correct.prod(-1)


def exact_pair_credit(behavior, targets, pairs, pair_weight):
    """E[R_pair | action_i] - E[R_pair], detached exact binary support."""
    validate(behavior,targets,pairs,pair_weight)
    if behavior.requires_grad:
        raise ValueError('detached behavior required')
    probability = behavior.exp()
    p_gold = probability.gather(1,targets[:,None]).squeeze(1)
    partner = torch.empty_like(targets)
    partner[pairs[:,0]]=pairs[:,1];partner[pairs[:,1]]=pairs[:,0]
    weight = (1-pair_weight)/2 + pair_weight*p_gold[partner]
    correctness = torch.nn.functional.one_hot(targets,2).to(behavior)
    return (weight[:,None]*(correctness-p_gold[:,None])).detach()


def pair_surrogate(behavior, actor, targets, pairs, pair_weight):
    """Unclipped diagnostic surrogate only; not a production verl loss.

    Mean over pairs of the sum of both exact conditional score contributions.
    Its on-policy gradient matches negative expected_pair_reward.mean().
    """
    validate(actor,targets,pairs,pair_weight)
    advantages = exact_pair_credit(behavior,targets,pairs,pair_weight)
    return -(behavior.exp()*advantages*(actor-behavior).exp()).sum()/len(pairs)

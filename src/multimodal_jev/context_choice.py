"""Open/closed audio Choice: input construction, frozen rewards and typed readouts."""
from dataclasses import replace
from hashlib import sha256
import math

import torch
import torch.nn.functional as F
from .context_decision_contract import ApplicabilityQuestion

COVER = '__list_covered__'


def shortlist(scores, ids, uid, spec):
    """Candidate proposal sees only frozen first-pass scores, ids and request seed."""
    if len(scores) != len(ids) or not all(map(math.isfinite, scores)) or len(set(ids)) != len(ids):
        raise ValueError('invalid retrieval scores or IDs')
    k = spec['shortlist_size']
    if not 0 < k <= len(ids) or not 0 <= spec['shortlist_dropout'] < 1:
        raise ValueError('invalid shortlist geometry')
    def key(role, i):
        return sha256(f"{spec['seed']}:{role}:{uid}:{ids[i]}".encode()).digest()
    ranked = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))
    kept = [i for i in ranked if int.from_bytes(key('retain', i)[:8], 'big') / 2**64 >= spec['shortlist_dropout']]
    selected = (kept + [i for i in ranked if i not in kept])[:k]
    return sorted(selected, key=lambda i: key('display', i))


def lengths(spec, n):
    values = spec['candidate_lengths']
    if not values or any(v != 'all' and (type(v) is not int or v < 1) for v in values):
        raise ValueError('positive candidate lengths or all required')
    return sorted({n if v == 'all' else min(v, n) for v in values})


def varied_slate(scores, ids, uid, spec, k, exposure):
    """Exposure and list width are fixed before binding ground truth."""
    key = f"{uid}:{exposure}"
    profiles = spec['proposal_profiles']
    if not profiles or not all(p in ('retrieved','dropout','uniform') for p in profiles):
        raise ValueError('unknown proposal profile')
    idx = int.from_bytes(sha256(f"{spec['seed']}:{key}:profile".encode()).digest()[:8], 'big')
    profile = profiles[idx % len(profiles)]
    if profile == 'uniform':
        scores = [int.from_bytes(sha256(f"{spec['seed']}:{key}:random:{tag}".encode()).digest()[:8],'big')
                  for tag in ids]
    options = dict(spec, shortlist_size=k, shortlist_dropout=spec['shortlist_dropout'] if profile=='dropout' else 0.)
    return shortlist(scores, ids, key, options), profile


def request(native, mode, indices):
    if not indices or len(set(indices)) != len(indices) or any(not 0 <= i < len(native.questions) for i in indices):
        raise ValueError('invalid slate')
    if mode == 'closed' and set(indices) != set(range(len(native.questions))):
        raise ValueError('closed benchmark requires complete native ontology')
    if mode not in ('closed', 'open') or COVER in {q.id for q in native.questions}:
        raise ValueError('invalid Choice contract')
    if native.version=='jev-semantic-answer-v1':
        qs=tuple(native.questions[i] for i in indices)
        rule='Select the listed answer that best answers the supplied question.'
        if mode=='open':
            rule+=' Decide whether the correct answer is in this list.'
            qs+=(ApplicabilityQuestion(COVER,'Coverage of the answer options.',
                                       'Does the list contain the correct answer to the supplied question?'),)
        return replace(native,questions=qs,requirements=(*native.requirements,rule),version='jev-semantic-choice-'+mode+'-v1')
    qs = tuple(replace(native.questions[i], question='Which listed intent best matches the audio?')
               for i in indices)
    rule = 'Exactly one of the listed intents is correct.'
    if mode == 'open':
        rule = 'The intended event may be absent. Judge list coverage and rank only the supplied intents.'
        qs += (ApplicabilityQuestion(COVER, 'Coverage of the listed intents, not an additional intent.',
                                    'Does the list contain the assigned intent?'),)
    return replace(native, questions=qs, requirements=(rule,), version='jev-choice-'+mode+'-v1')


def distribution(z, mode, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('positive readout temperature required')
    if mode == 'closed':
        return (z.float()/temperature).log_softmax(-1)
    if mode != 'open' or z.shape[-1] < 2:
        raise ValueError('candidate(s) plus coverage marker required')
    g = z[..., -1:].float()
    return torch.cat((F.logsigmoid(g)+(z[..., :-1].float()/temperature).log_softmax(-1),
                      F.logsigmoid(-g)), -1)


def projection(z, ref, gold, mode, spec):
    """Misses have zero conditional-ranking gradient; g=coverage is its own action."""
    t, b = spec['temperature'], spec['reward_gain']/spec['beta']
    if not math.isfinite(b) or b <= 0:
        raise ValueError('positive reward scale required')
    if mode not in ('closed','open') or z.shape != ref.shape:
        raise ValueError('target geometry')
    logits = z.float() if mode == 'closed' else z[:, :-1].float()
    prior = ref.detach().float() if mode == 'closed' else ref[:, :-1].detach().float()
    hit = gold >= 0
    if (gold >= logits.size(-1)).any() or (gold < -1).any() or (mode == 'closed' and not hit.all()):
        raise ValueError('invalid supervised choice')
    bonus = F.one_hot(gold.clamp_min(0), logits.size(-1)).float()*hit[:,None]*b
    logq = (prior/t + bonus).log_softmax(-1)
    rank = (logq.exp()*(logq - (logits/t).log_softmax(-1))).sum(-1)*hit
    if mode == 'closed':
        return rank
    from .context_tpo_rewards import projection as binary
    return rank + binary(z[:,-1:], ref[:,-1:].detach() + (2*hit.float()-1)[:,None]*b)

"""Native attribute reward/Final auxiliary candidate, never a training entrypoint.

Parallel families are label projections, not time steps. The caller authenticates
the catalog and policy and computes full-group advantages before DP partitioning.
"""
import torch
from multimodal_jev.context_set_reward import set_f1
from multimodal_jev.context_candidate_objective import loss as candidate_auxiliary


def layout(menu, arm):
    if arm not in ('direct', 'q_input', 'q_credit'):
        raise ValueError('Known arm required')
    events = [e['id'] for e in menu['events']]
    nodes = menu['nodes']
    if (not events or len(set(events)) != len(events) or not nodes
            or len({n['id'] for n in nodes}) != len(nodes)
            or any(n['family'] not in ('scenario','action','final') for n in nodes)):
        raise ValueError('Distinct events/nodes and known families required')
    for family in ('scenario','action','final'):
        family_nodes = [n for n in nodes if n['family'] == family]
        members = [m for n in family_nodes for m in n['members']]
        if not family_nodes or sorted(members) != sorted(events):
            raise ValueError('Every family must partition the event catalog exactly once')
        if family == 'final' and any(len(n['members']) != 1 for n in family_nodes):
            raise ValueError('Final nodes must be singletons')
    active = [n for n in nodes if arm != 'direct' or n['family'] == 'final']
    return active, {f:[i for i,n in enumerate(active) if n['family'] == f]
                    for f in ('scenario','action','final')}


def validate_truth(labels, observed, menu, arm):
    active, indices = layout(menu, arm)
    if (labels.ndim != 2 or labels.shape[0] < 1 or labels.shape[1] != len(active)
            or labels.dtype != torch.bool or observed.dtype != torch.bool
            or observed.shape != labels.shape or observed.device != labels.device
            or not observed.all()):
        raise ValueError('Complete Boolean native truth required; missing is not false')
    final = indices['final']
    if not (labels[:,final].sum(-1) == 1).all():
        raise ValueError('Exactly one assigned Final event required')
    # Independent projection check prevents inconsistent yet singleton families.
    projection = torch.tensor([[n['members'][0] in a['members'] for a in active]
                              for n in (active[i] for i in final)],
                             device=labels.device,dtype=torch.bool)
    gold = labels[:,final].long().argmax(-1)
    if not torch.equal(labels, projection[gold]):
        raise ValueError('Attribute truth must match assigned Final event membership')
    return indices


def rewards(actions, labels, observed, menu, *, arm):
    indices = validate_truth(labels, observed, menu, arm)
    if (actions.ndim != 3 or actions.shape[0] != labels.shape[0]
            or actions.shape[2] != labels.shape[1] or actions.shape[1] < 2
            or actions.dtype != torch.long or actions.device != labels.device
            or ((actions != 0) & (actions != 1)).any()):
        raise ValueError('Complete binary action groups G>=2 required')
    scores = {f:set_f1(actions[:,:,idx],labels[:,idx],observed[:,idx])
              for f,idx in indices.items() if idx}
    total = scores['final']
    if arm == 'q_credit':
        total = total + .25*(scores['scenario']+scores['action'])/2
    # Reward diagnostics never repair Final predictions from the attributes.
    return total.detach(), scores


def final_auxiliary(logits, labels, observed, request_weights, real_requests,
                    menu, *, arm, **kwargs):
    indices = validate_truth(labels, observed, menu, arm)['final']
    if logits.shape != labels.shape or logits.device != labels.device:
        raise ValueError('Aligned logits required')
    return candidate_auxiliary(logits[:,indices],labels[:,indices],observed[:,indices],
        request_weights,real_requests,**kwargs)

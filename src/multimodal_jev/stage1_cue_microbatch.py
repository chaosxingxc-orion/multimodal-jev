"""Stage1 candidate: native all-row loss plus local credit on verified paired recordings.

Optimizer-side supervision only. Both members of an auxiliary unit must occur
in the same microbatch. Do not silently drop a split unit.
"""
import math

import torch
from torch.nn import functional as F

from .aligned_cue_objective import cue_pair_loss
from .context_infonce import loss as native_loss, validate_objective


def compute(logits, labels, observed, weights, real, *, records, units, candidates,
            objective, global_real_requests, world_size=1):
    """records gives (source ID, audio SHA) in actual row order; units is the complete index by ID."""
    native = objective['native']
    validate_objective(native)
    if (native['kind'] != 'candidate_infonce_probability_margin_v1'
            or set(objective) != {'native','flip_weight','preserve_weight'}
            or any(type(objective[k]) not in (int,float) or not math.isfinite(objective[k])
                   or objective[k] < 0 for k in ('flip_weight','preserve_weight'))):
        raise ValueError('explicit native and finite nonnegative cue weights required')
    value, stats = native_loss(logits, labels, observed, weights, real, temperature=native['temperature'],
        probability_margin=native['probability_margin'], global_real_requests=global_real_requests,
        world_size=world_size)
    if (len(records) != logits.shape[0] or len(candidates) != logits.shape[1]
            or len(set(candidates)) != len(candidates)):
        raise ValueError('one recording per row and unique semantic candidate IDs required')
    local = {records[i][0]:i for i in range(len(records)) if bool(real[i])}
    if len(local) != int(real.sum()):
        raise ValueError('duplicate real source recording')
    seen=set()
    sums = {k:logits.float().sum()*0 for k in ('flip','preserve')}
    counts = dict(flip=0,preserve=0)
    for uid in local:
        if uid in seen or uid not in units:
            continue
        u = units[uid]
        a,b = u['rows']
        ids = [a['id'], b['id']]
        if (len(set(ids)) != 2 or uid not in ids or not all(k in local and units.get(k)==u for k in ids)
                or a['audio_sha256'] == b['audio_sha256']):
            raise ValueError('complete disjoint pair required on one rank and microbatch')
        i,j = [local[k] for k in ids]
        family = u['family']
        if (len(set(family))!=2 or len(family)!=2 or not set(family)<=set(candidates)
                or any(r['gold'] not in family or records[n][1]!=r['audio_sha256']
                       or not bool(labels[n,candidates.index(r['gold'])]) for n,r in ((i,a),(j,b)))):
            raise ValueError('pair audio or semantic binding differs')
        kind=u['kind']
        if kind != ('flip' if a['gold']!=b['gold'] else 'preserve'):
            raise ValueError('invalid cue relation')
        target = candidates.index(a['gold'])
        foil = candidates.index(next(k for k in family if k!=a['gold']))
        if kind=='flip':
            ind=lambda n: torch.tensor([n],device=logits.device)
            term=cue_pair_loss(logits[i:i+1],logits[j:j+1],ind(target),ind(foil),
                temperature=native['temperature'],probability_margin=native['probability_margin']['margin'])[0]
        else:
            # Enforce stability of the supervised margin; native supervision anchors both sides.
            p=logits[[i,j]].float().sigmoid()
            gap=p[:,target]-p[:,foil]
            term=F.smooth_l1_loss(gap[0],gap[1])
        sums[kind]=sums[kind]+2*term
        counts[kind]+=2
        seen.update(ids)
    for kind in counts:
        value=value+objective[kind+'_weight']*sums[kind]*world_size/global_real_requests
        stats[kind+'_rows']=counts[kind]
        stats[kind+'_numerator']=float(sums[kind].detach())
    stats['total_numerator']+=sum(objective[k+'_weight']*stats[k+'_numerator'] for k in counts)
    return value,stats

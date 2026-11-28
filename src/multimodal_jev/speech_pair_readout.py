"""Counts from the same A/P/N inputs at the same weights, before the optimizer update."""
from .speech_alignment_pairs import KINDS

FIELDS = ('panels','speech_correct','cards_invariant','cards_switched','cards_both','both_correct')


def counter_names():
    return tuple('jev/pair_'+k+'_'+f for k in KINDS for f in FIELDS)


def extract(raw, vectors, groups, d):
    import torch
    from torch.nn import functional as F
    offsets, panels, kinds, rivals, foils = [], [], [], [], []
    start = 0
    for rows,weight,_,_,kind in groups:
        card, speech = 1, d.anchor_end
        if isinstance(kind, tuple):
            kind, card, speech = kind
            if not 1 <= card < d.anchor_end or not d.anchor_end <= speech < d.observed:
                raise ValueError('semantic control outside the negative slots')
        if len(rows)!=d.width + (2 if kind else 0) or (kind is not None and kind not in KINDS):
            raise ValueError('incomplete semantic panel')
        offsets.extend(range(start,start+d.width))
        if kind and weight:
            panels.append(start); kinds.append(kind)
            rivals.append(start+card); foils.append(start+speech)
        start+=len(rows)
    if raw.numel()!=start or vectors.ndim!=2 or vectors.size(0)!=start:
        raise ValueError('speech and membership must cover exactly the same inputs')
    with torch.no_grad():
        ix=torch.tensor(panels,device=raw.device,dtype=torch.long)
        z=raw.detach().float().flatten().sigmoid()  # Match the deployed probability, including FP32 ties.
        a,p=ix,ix+d.observed
        h=torch.tensor(rivals,device=raw.device,dtype=torch.long)
        n=torch.tensor(foils,device=raw.device,dtype=torch.long)
        # Four independent scores on the two competing cards, plus both scores on P.
        switch=(z[a]>z[h]) & (z[ix+d.width]>z[n])
        invariant=(z[a]>z[h]) & (z[p]>z[ix+d.width+1])
        v=vectors.detach().float()
        speech=F.cosine_similarity(v[a],v[p])>F.cosine_similarity(v[a],v[n])
        cards = invariant & switch
        flags=torch.stack((torch.ones_like(switch),speech,invariant,switch,cards,speech & cards),dim=1)
        masks=torch.tensor([[k==kind for k in kinds] for kind in KINDS],device=z.device,dtype=torch.bool)
        counts=(masks[:,:,None] & flags[None,:,:]).sum(1).flatten()
    return torch.tensor(offsets,device=raw.device,dtype=torch.long),counts


def tally(metrics, real):
    result = {}
    for k in KINDS:
        row = {f:metrics['jev/pair_'+k+'_'+f] for f in FIELDS}
        if any(type(n) is not int or not 0<=n<=row['panels'] for n in row.values()):
            raise ValueError('invalid semantic pair count')
        for a,b,joint in [('cards_invariant','cards_switched','cards_both'),
                          ('speech_correct','cards_both','both_correct')]:
            if not max(0,row[a]+row[b]-row['panels']) <= row[joint] <= min(row[a],row[b]):
                raise ValueError('invalid intersection of paired successes')
        result[k]=row
    if sum(r['panels'] for r in result.values())>real:
        raise ValueError('semantic counts include padding or duplicate anchors')
    return result

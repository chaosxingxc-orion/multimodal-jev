"""Label-free 2D length buckets and balanced fixed-size DP microbatches.

Changes only permutation inside an optimizer batch. Fixed micro sizes retain
native verl accumulation/loss normalization and zero-weight tail semantics.
"""
from .span_length_groups import length_group_indices


def cost(group,lengths,weights):
    tokens=max(lengths[i][0] for i in group)
    frames=max(lengths[i][1] for i in group)
    return len(group)*(weights[0]*tokens+weights[1]*frames)


def metrics(ranks,lengths,microbatch,weights):
    groups=[[r[i:i+microbatch] for i in range(0,len(r),microbatch)] for r in ranks]
    totals=[sum(cost(g,lengths,weights) for g in gs) for gs in groups]
    return dict(padded_tokens=sum(len(g)*max(lengths[i][0] for i in g) for gs in groups for g in gs),
        padded_frames=sum(len(g)*max(lengths[i][1] for i in g) for gs in groups for g in gs),
        rank_cost_proxy=totals,slowest_rank_cost_proxy=max(totals),
        total_cost_proxy=sum(totals),scope='padding/allocation proxy, not measured device runtime')


def plan(lengths,*,microbatch,world_size,token_bucket_width=128,frame_bucket_width=200,
         token_cost=1,frame_cost=1):
    baseline=length_group_indices(lengths,microbatch=microbatch,world_size=world_size,order='length_grouped')
    if any(type(v) is not int or v<1 for v in (token_bucket_width,frame_bucket_width,token_cost,frame_cost)):
        raise ValueError('positive integer bucket widths and cost weights required')
    weights=(token_cost,frame_cost); local=len(lengths)//world_size
    orders={
        'token_then_frame': sorted(range(len(lengths)),key=lambda i:(*lengths[i],i)),
        'frame_then_token': sorted(range(len(lengths)),key=lambda i:(lengths[i][1],lengths[i][0],i)),
        'token_bucket_frame': sorted(range(len(lengths)),key=lambda i:(lengths[i][0]//token_bucket_width,lengths[i][1],lengths[i][0],i)),
        'frame_bucket_token': sorted(range(len(lengths)),key=lambda i:(lengths[i][1]//frame_bucket_width,lengths[i][0],lengths[i][1],i)),
    }
    candidates=[('baseline',baseline,metrics(baseline,lengths,microbatch,weights))]
    for name,order in orders.items():
        groups=[order[i:i+microbatch] for i in range(0,len(order),microbatch)]
        groups.sort(key=lambda g:(-cost(g,lengths,weights),g[0]))
        ranks=[[] for _ in range(world_size)];loads=[0]*world_size
        for group in groups:
            rank=min((r for r in range(world_size) if len(ranks[r])<local),key=lambda r:(loads[r],r))
            ranks[rank].extend(group);loads[rank]+=cost(group,lengths,weights)
        candidates.append((name,ranks,metrics(ranks,lengths,microbatch,weights)))
    name,ranks,report=min(candidates,key=lambda row:(row[2]['slowest_rank_cost_proxy'],row[2]['total_cost_proxy']))
    assert sorted(i for rank in ranks for i in rank)==list(range(len(lengths)))
    assert all(len(rank)==local for rank in ranks)
    return ranks,dict(selected=name,baseline=candidates[0][2],candidate=report,
        settings=dict(token_bucket_width=token_bucket_width,frame_bucket_width=frame_bucket_width,
                      token_cost=token_cost,frame_cost=frame_cost,microbatch=microbatch,world_size=world_size))

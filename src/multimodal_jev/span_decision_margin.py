"""Full-support gold rank and decision margin, without threshold selection."""
import math
from statistics import median


def measure(logprobs,target):
    if (type(target) is not int or not 0<=target<len(logprobs) or len(logprobs)<2
            or any(type(v) not in (int,float) or not math.isfinite(v) for v in logprobs)
            or abs(sum(math.exp(v) for v in logprobs)-1)>1e-5):
        raise ValueError('normalized complete candidate probabilities and gold required')
    ordered=sorted(range(len(logprobs)),key=lambda i:-logprobs[i])
    margin=logprobs[target]-max(v for i,v in enumerate(logprobs) if i!=target)
    return dict(gold_rank=ordered.index(target)+1,gold_log_margin=margin,
        tied_gold=sum(v==logprobs[target] for v in logprobs)>1,
        correct=ordered[0]==target,top3=target in ordered[:3],top10=target in ordered[:10])


def summarize(records):
    if not records or len({r['id'] for r in records})!=len(records):raise ValueError('unique fixed population required')
    return dict(recordings=len(records),correct=sum(r['correct'] for r in records),
        top3_correct=sum(r['top3'] for r in records),top10_correct=sum(r['top10'] for r in records),
        mean_reciprocal_rank=sum(1/r['gold_rank'] for r in records)/len(records),
        median_gold_rank=median(r['gold_rank'] for r in records),
        median_gold_log_margin=median(r['gold_log_margin'] for r in records),
        positive_margin=sum(r['gold_log_margin']>0 for r in records),
        zero_margin=sum(r['gold_log_margin']==0 for r in records),
        tie_policy='first supplied candidate; margin0 alone does not imply correct',
        scope='Descriptive fixed-learning candidate ranking; top3/top10 are not the Final exact primary metric.')

"""Descriptive variation of complete categorical policies over fixed inputs.

No labels, calibration, causal audio attribution, or probability correction.
"""
import math
from collections import Counter


def summarize(logprobs, values, symbols):
    if (not logprobs or len(values)<2 or len(set(values))!=len(values)
            or len(symbols)!=len(values) or len(set(symbols))!=len(symbols)):
        raise ValueError('nonempty distinct fixed support required')
    rows=[]
    for row in logprobs:
        if len(row)!=len(values) or any(type(x) not in (int,float) or not math.isfinite(x) for x in row):
            raise ValueError('finite full-support policy required')
        p=[math.exp(x) for x in row];mass=sum(p)
        if abs(mass-1)>1e-5:raise ValueError('normalized categorical probabilities required')
        rows.append([v/mass for v in p])
    n=len(rows);k=len(values);mean=[sum(p[i] for p in rows)/n for i in range(k)]
    entropy=lambda p:-sum(v*math.log(v) for v in p if v>0)
    mean_h=sum(entropy(p) for p in rows)/n;pooled_h=entropy(mean)
    js=sum(sum(v*math.log(v/mean[i]) for i,v in enumerate(p) if v>0) for p in rows)/n
    predictions=Counter(max(range(k),key=p.__getitem__) for p in rows)
    top=sorted(range(k),key=lambda i:-mean[i]);groups={}
    for name, predicate in (('uppercase',str.isupper),('lowercase',str.islower),('digit',str.isdigit)):
        indices=[i for i,s in enumerate(symbols) if predicate(s)]
        if indices:
            mass=sum(mean[i] for i in indices)
            groups[name]=dict(candidates=len(indices),pooled_mass=mass,uniform_mass=len(indices)/k,
                mass_over_uniform=mass/(len(indices)/k))
    return dict(requests=n,candidates=k,mean_request_entropy=mean_h,pooled_entropy=pooled_h,
        mean_kl_request_to_pooled=js,entropy_decomposition_residual=pooled_h-mean_h-js,
        mean_total_variation_to_pooled=sum(sum(abs(a-b) for a,b in zip(p,mean))/2 for p in rows)/n,
        predicted_classes=len(predictions),dominant_prediction_fraction=max(predictions.values())/n,
        pooled_top1_mass=mean[top[0]],pooled_top3_mass=sum(mean[i] for i in top[:3]),
        symbol_groups=groups,candidates_by_pooled_mass=[dict(value=values[i],symbol=symbols[i],
            pooled_probability=mean[i],prediction_count=predictions[i]) for i in top],
        scope='Descriptive input-population variation. Fixed symbol/label binding confounds semantic and symbol preferences; '
              'low or high variation alone does not prove audio use, semantic discrimination or label accuracy.')

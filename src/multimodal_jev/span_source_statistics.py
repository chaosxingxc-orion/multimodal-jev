"""Paired bootstrap at audio source-group level, never individual rule rows."""
from collections import defaultdict
import random


def paired_source_interval(rows,before,after,*,replicates=2000,seed=17):
    if type(replicates) is not int or replicates<100 or type(seed) is not int:
        raise ValueError('explicit reproducible bootstrap budget required')
    ids=[r['id'] for r in rows];a={r['id']:r for r in before};b={r['id']:r for r in after}
    if (not rows or len(ids)!=len(set(ids)) or len(a)!=len(before) or len(b)!=len(after)
            or set(ids)!=a.keys() or a.keys()!=b.keys()):
        raise ValueError('complete paired source population required')
    groups=defaultdict(lambda:defaultdict(list))
    for row in rows:
        key=row['id'];gold=row['targets'][-1]
        if type(gold) is not bool or not row['supervised'][-1]:raise ValueError('reviewed Boolean Final required')
        for pred in (a[key],b[key]):
            if len(pred['predictions'])!=len(row['targets']) or any(type(v) is not bool for v in pred['predictions']):
                raise ValueError('aligned Boolean predictions required')
        source=row['provenance']['source_group']
        groups[row['arm']][source].append((int(a[key]['predictions'][-1]==gold),int(b[key]['predictions'][-1]==gold)))
    result={}
    for arm,sources in sorted(groups.items()):
        if len(sources)<2:raise ValueError('at least two independent source groups required')
        aggregates=[(sum(x for x,y in items),sum(y for x,y in items),len(items)) for _,items in sorted(sources.items())]
        old=sum(x for x,y,n in aggregates);new=sum(y for x,y,n in aggregates);n=sum(n for x,y,n in aggregates)
        rng=random.Random(seed);micro=[];macro=[]
        for _ in range(replicates):
            sample=[aggregates[rng.randrange(len(aggregates))] for _ in aggregates]
            micro.append(sum(y-x for x,y,n in sample)/sum(n for x,y,n in sample))
            macro.append(sum((y-x)/n for x,y,n in sample)/len(sample))
        def interval(values):
            values=sorted(values)
            def quantile(q):
                index=(len(values)-1)*q;lo=int(index);hi=min(lo+1,len(values)-1)
                return values[lo]+(values[hi]-values[lo])*(index-lo)
            return [quantile(.025),quantile(.975)]
        result[arm]=dict(requests=n,source_groups=len(sources),before_accuracy=old/n,after_accuracy=new/n,
            request_weighted_delta=(new-old)/n,request_weighted_delta_ci95=interval(micro),
            source_macro_delta=sum((y-x)/n for x,y,n in aggregates)/len(aggregates),source_macro_delta_ci95=interval(macro))
    return dict(arms=result,replicates=replicates,seed=seed,
        unit='source_group; all correlated rules and complements sampled together',
        scope='Paired descriptive percentile intervals for this fixed policy pair and sampled population. Not multi-seed uncertainty, speaker isolation, natural-distribution or causal-gain evidence; no multiple-comparison correction.')

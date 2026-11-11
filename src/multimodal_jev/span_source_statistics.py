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


def paired_rule_ba_interval(rows,before,after,*,replicates=2000,seed=17):
    """Source-cluster bootstrap for the prespecified macro-rule BA metric.

    A replicate lacking either label for any rule has undefined full macro BA.
    Report that count and exclude it only from the descriptive percentile
    interval, never average over a smaller, easier set of rules.
    """
    import numpy as np
    # Reuse strict complete/Boolean validation, without another full bootstrap.
    paired_source_interval(rows,before,after,replicates=100,seed=seed)
    if type(replicates) is not int or replicates<100:raise ValueError('explicit bootstrap budget required')
    a={r['id']:r for r in before};b={r['id']:r for r in after};result={}
    for arm in sorted({r['arm'] for r in rows}):
        selected=[r for r in rows if r['arm']==arm]
        sources=sorted({r['provenance']['source_group'] for r in selected})
        rules=sorted({r['provenance']['rule_id'] for r in selected})
        si={s:i for i,s in enumerate(sources)};ri={r:i for i,r in enumerate(rules)}
        counts=np.zeros((len(sources),len(rules),2,3),dtype=np.int64)
        for row in selected:
            uid=row['id'];g=row['targets'][-1];p=row['provenance']
            counts[si[p['source_group']],ri[p['rule_id']],int(g)]+=[1,int(a[uid]['predictions'][-1]==g),int(b[uid]['predictions'][-1]==g)]
        full=counts.sum(0)
        if np.any(full[:,:,0]==0):raise ValueError('all prespecified rules require both gold labels')
        old=float((full[:,:,1]/full[:,:,0]).mean());new=float((full[:,:,2]/full[:,:,0]).mean())
        rng=random.Random(seed);deltas=[]
        for _ in range(replicates):
            sampled=counts[[rng.randrange(len(sources)) for _ in sources]].sum(0)
            if np.any(sampled[:,:,0]==0):continue
            deltas.append(float(((sampled[:,:,2]-sampled[:,:,1])/sampled[:,:,0]).mean()))
        result[arm]=dict(source_groups=len(sources),rules=len(rules),before_macro_rule_ba=old,
            after_macro_rule_ba=new,delta_macro_rule_ba=new-old,valid_replicates=len(deltas),
            undefined_replicates=replicates-len(deltas),
            delta_ci95=None if len(deltas)<100 else np.quantile(deltas,[.025,.975]).tolist())
    return dict(arms=result,replicates=replicates,seed=seed,
        unit='source_group; complete rule and label roster retained in every valid replicate',
        scope='Descriptive source-bootstrap macro-rule BA difference; interval conditional on full label coverage. No seed uncertainty, multiplicity correction or causal claim.')

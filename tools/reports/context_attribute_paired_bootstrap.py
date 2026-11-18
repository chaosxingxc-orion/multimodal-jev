"""Paired source-group bootstrap for frozen endpoints, not training variance.

Resample whole source groups with replacement, retaining all recordings and
their paired differences. The estimand remains recording-weighted mean gain.
Selected endpoints are conditional descriptions, not selection-adjusted CIs.
"""
import hashlib
import json
import math
import numpy as np

FIELDS=('set_f1','exact_set','top1_correct')
MATCH=('id','source_group','gold','structural_slice')


def aggregate(a,b):
    if not a or len(a)!=len(b) or len({r['id'] for r in a})!=len(a):
        raise ValueError('Complete unique paired recording population required')
    grouped={};truth={}
    # Canonical ID order makes results independent of file order, while requiring
    # each supplied pair to refer to the same actual recording and source group.
    for x,y in sorted(zip(a,b,strict=True),key=lambda pair:pair[0]['id']):
        if any(x[k]!=y[k] for k in MATCH):raise ValueError('Pairing or source identity differs')
        for r in (x,y):
            if (any(type(r[k]) is not str or not r[k] for k in MATCH)
                    or type(r['set_f1']) not in (float,int) or not math.isfinite(r['set_f1'])
                    or not 0<=r['set_f1']<=1 or type(r['exact_set']) is not bool
                    or type(r['top1_correct']) is not bool):
                raise ValueError('Finite bounded scores and Boolean decisions required')
        group=x['source_group'];label=(x['gold'],x['structural_slice'])
        if truth.setdefault(group,label)!=label:raise ValueError('Conflicting source group truth or slice')
        values=grouped.setdefault(group,[])
        values.append([float(y[k])-float(x[k]) for k in FIELDS])
    names=sorted(grouped)
    if len(names)<2:raise ValueError('At least two source groups required')
    sums=np.asarray([[math.fsum(row[i] for row in grouped[g]) for i in range(len(FIELDS))] for g in names])
    counts=np.asarray([len(grouped[g]) for g in names],dtype=np.int64)
    return names,sums,counts


def replicate_means(sums,counts,draws):
    """One draw contains K cluster indices; unequal cluster sizes are retained."""
    if (sums.ndim!=2 or sums.shape[1]!=len(FIELDS) or counts.shape!=(len(sums),)
            or not np.isfinite(sums).all() or (counts<=0).any()
            or draws.ndim!=2 or draws.shape[1]!=len(counts)
            or draws.dtype.kind not in 'iu' or draws.size==0
            or (draws<0).any() or (draws>=len(counts)).any()):
        raise ValueError('Complete finite cluster aggregates and resampling indices required')
    return sums[draws].sum(axis=1)/counts[draws].sum(axis=1)[:,None]


def compare_records(a,b,*,endpoint):
    if endpoint not in ('fixed_step634','source_selected_conditional'):
        raise ValueError('Predeclared fixed endpoint or conditional selected endpoint required')
    names,sums,counts=aggregate(a,b)
    seed=317;replicates=2000
    rng=np.random.Generator(np.random.PCG64(seed))
    draws=rng.integers(0,len(names),size=(replicates,len(names)),dtype=np.int64)
    # Bound temporary storage for the full population while retaining exact draws.
    values=np.concatenate([replicate_means(sums,counts,d) for d in np.array_split(draws,20)])
    point=sums.sum(axis=0)/counts.sum()
    intervals=np.quantile(values,[.025,.975],axis=0,method='linear')
    pairing=[{k:r[k] for k in MATCH} for r in sorted(a,key=lambda r:r['id'])]
    return dict(endpoint=endpoint,recordings=len(a),source_groups=len(names),
        pairing_sha256=hashlib.sha256(json.dumps(pairing,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        metrics={k:dict(delta=float(point[i]),percentile_95=[float(v) for v in intervals[:,i]]) for i,k in enumerate(FIELDS)},
        bootstrap=dict(unit='source_group',paired=True,replicates=replicates,seed=seed,rng='PCG64',
            quantile_method='linear',estimand='recording_weighted_mean_delta',
            draws_sha256=hashlib.sha256(draws.astype('<i8',copy=False).tobytes()).hexdigest()),
        scope='Conditional on fixed model outputs and source-group sampling assumptions. Not speaker-disjoint inference, training-seed variance, multiplicity correction or selection-adjusted coverage. No checkpoint selection by interval.')

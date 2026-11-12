"""Audit source/class coefficients after actual per-batch normalization.

These are objective coefficients, not gradient shares or an Adam trajectory.
No scheduler, sampling rule or training weight is modified.
"""
from collections import Counter
import math


def audit(sources,batches):
    rows={r['id']:r for r in sources}
    if not rows or len(rows)!=len(sources) or not batches:
        raise ValueError('distinct sources and nonempty schedule required')
    for row in sources:
        if (not isinstance(row['label'],str) or not row['label']
                or type(row['weight']) not in (int,float)
                or not math.isfinite(row['weight']) or row['weight']<=0):
            raise ValueError('named classes and positive finite weights required')
    counts=Counter(r['label'] for r in sources);n=len(sources);classes=len(counts)
    for row in sources:
        if not math.isclose(row['weight'],n/(classes*counts[row['label']]),rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('original inverse-frequency population weights required')
    mass=Counter();raw=Counter();uses=Counter();class_uses=Counter();reports=[]
    for index,batch in enumerate(batches):
        if not batch or len(batch)!=len(set(batch)) or not set(batch)<=rows.keys():
            raise ValueError('distinct known sources in every batch required')
        denominator=sum(rows[i]['weight'] for i in batch);local=Counter()
        for uid in batch:
            row=rows[uid];label=row['label'];uses[uid]+=1;class_uses[label]+=1
            local[label]+=row['weight']/denominator
            mass[label]+=row['weight']/denominator/len(batches)
            raw[label]+=row['weight']
        reports.append(dict(update=index+1,recordings=len(batch),classes=len(local),
            weight_denominator=denominator,class_coefficients=dict(sorted(local.items()))))
    total_raw=sum(raw.values());ideal=1/classes
    labels=sorted(counts);values=[mass[k] for k in labels]
    return dict(recordings=n,observed_classes=classes,updates=len(batches),source_uses=sum(uses.values()),
        source_use_histogram=dict(sorted(Counter(uses[i] for i in rows).items())),
        population_class_coefficient=ideal,
        scheduled_class_coefficient_range=[min(values),max(values)],
        scheduled_max_to_min_ratio=max(values)/min(values) if min(values)>0 else None,
        total_variation_from_uniform=.5*sum(abs(v-ideal) for v in values),
        pooled_weight_total_variation_from_uniform=.5*sum(abs(raw[k]/total_raw-ideal) for k in labels),
        per_class=[dict(label=k,population_sources=counts[k],scheduled_uses=class_uses[k],
            scheduled_coefficient=mass[k],pooled_weight_coefficient=raw[k]/total_raw,
            deviation_from_uniform=mass[k]-ideal) for k in labels],batches=reports,
        scope='Equal average of registered batch-normalized objective coefficients at a common hypothetical policy. '
              'Not gradient attribution, optimizer displacement, exact class-balanced batches or a learning result. '
              'Pooled weights isolate repeat-source imbalance, not a proposed training change.')

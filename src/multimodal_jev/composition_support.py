"""Ontology ambiguity and prospective held-out-combination feasibility.

Gold-component oracles measure label structure, not deployable model quality.
No new supervision or train/evaluation admission is granted by this audit.
"""
from collections import Counter,defaultdict
import hashlib
from itertools import combinations


def audit(mapping,counts=None,*,seed=941):
    if not mapping or type(seed) is not int:raise ValueError('nonempty ontology and integer seed required')
    rows={k:tuple(v) for k,v in mapping.items()}
    if any(type(k) is not str or not k for k in rows):raise ValueError('named labels required')
    width=len(next(iter(rows.values())))
    if (not 2<=width<=8 or any(len(v)!=width or any(type(x) is not str or not x for x in v) for v in rows.values())
            or len(set(rows.values()))!=len(rows)):
        raise ValueError('distinct complete component tuples required')
    if counts is not None and (set(counts)-set(rows) or any(type(n) is not int or n<0 for n in counts.values())):
        raise ValueError('nonnegative integer counts on known labels required')
    supports=[defaultdict(list) for _ in range(width)]
    for label,values in rows.items():
        for j,value in enumerate(values):supports[j][value].append(label)
    single_unique={label:[j for j,v in enumerate(values) if len(supports[j][v])==1] for label,values in rows.items()}
    all_ambiguous=sorted(k for k,v in single_unique.items() if not v)
    components=[]
    for j,support in enumerate(supports):
        report=dict(index=j,values=len(support),degrees={v:len(labels) for v,labels in sorted(support.items())},
            uniquely_identified_labels=sorted(k for k,v in single_unique.items() if j in v),
            uniform_label_oracle_accuracy=len(support)/len(rows))
        if counts is not None:
            total=sum(counts.values())
            report['learning_gold_component_majority_accuracy']=sum(max(counts.get(k,0) for k in labels) for labels in support.values())/total if total else None
            report['learning_unique_label_recordings']=sum(counts.get(k,0) for k in report['uniquely_identified_labels'])
        components.append(report)
    subset_reports=[];minimum={k:width for k in rows}
    for size in range(1,width):
        for indices in combinations(range(width),size):
            groups=defaultdict(list)
            for label,values in rows.items():groups[tuple(values[j] for j in indices)].append(label)
            unique=sorted(labels[0] for labels in groups.values() if len(labels)==1)
            for label in unique:minimum[label]=min(minimum[label],size)
            subset_reports.append(dict(indices=list(indices),distinct_values=len(groups),
                uniquely_identified_labels=unique,uniform_label_oracle_accuracy=len(groups)/len(rows),
                learning_gold_subset_majority_accuracy=(sum(max(counts.get(k,0) for k in labels) for labels in groups.values())/sum(counts.values())
                    if counts is not None and sum(counts.values()) else None)))
    # Prospective, deterministic feasible set only. Not maximum-cardinality;
    # every held-out label must have support, and each of its components must
    # remain represented by at least one retained supported label.
    active=set(rows) if counts is None else {k for k,n in counts.items() if n>0}
    remaining=set(active);heldout=[]
    degrees=[Counter(rows[k][j] for k in active) for j in range(width)]
    ordering=sorted(active,key=lambda k:(hashlib.sha256(f'{seed}:{k}'.encode()).hexdigest(),k))
    for label in ordering:
        values=rows[label]
        if all(degrees[j][v]>1 for j,v in enumerate(values)):
            remaining.remove(label);heldout.append(label)
            for j,v in enumerate(values):degrees[j][v]-=1
    witnesses={label:{str(j):sorted(k for k in remaining if rows[k][j]==value)
        for j,value in enumerate(rows[label])} for label in heldout}
    if any(not ids for row in witnesses.values() for ids in row.values()):raise AssertionError('unsupported held-out component')
    return dict(labels=len(rows),components=components,proper_subsets=subset_reports,
        minimum_identifying_components=minimum,
        requires_all_components_labels=sorted(k for k,n in minimum.items() if n==width),
        single_component_unique=single_unique,
        all_single_components_ambiguous_labels=all_ambiguous,
        all_single_components_ambiguous_recordings=sum(counts.get(k,0) for k in all_ambiguous) if counts is not None else None,
        learning_recordings=sum(counts.values()) if counts is not None else None,
        prospective_split=dict(seed=seed,method='hash-order greedy component-preserving removal; not optimal or an approved split',
            supported_labels=len(active),retained_labels=sorted(remaining),heldout_labels=sorted(heldout),witnesses=witnesses,
            heldout_recordings=sum(counts.get(k,0) for k in heldout) if counts is not None else None),
        admission=False,scope='Ontology/support audit. Gold-component oracles are in-population structural diagnostics, not audio inference or generalization.')

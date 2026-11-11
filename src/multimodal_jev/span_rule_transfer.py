"""Semantic rule holdout with separate source and task axes.

Canonicalize truth tables under input renaming and output complement. Input
negation is deliberately NOT a symmetry: it changes the supplied predicate.
This groups equivalent syntax, reversed arguments and R/NOT R together.
"""
import itertools
import random
from collections import defaultdict
from .decision_rules import dependencies, evaluate, Fact
from .span_rule_balance import balanced_rule_schedule


def semantic_rule_group(rule):
    keys=sorted(dependencies(rule))
    if len(keys)!=2:
        raise ValueError('exactly two Boolean dependencies required')
    definitions={k:k for k in keys}
    variants=[]
    for ordered in itertools.permutations(keys):
        table=[]
        for values in itertools.product((False,True),repeat=2):
            facts={k:Fact(v,'reviewed') for k,v in zip(ordered,values,strict=True)}
            table.append(int(evaluate(rule,definitions,facts)['value']))
        variants.extend((table,[1-v for v in table]))
    return 'bool2:'+''.join(map(str,min(variants)))


# Derive this from the predicate, not a surface name such as "except".
EXCLUSION_GROUP=semantic_rule_group({'op':'except','base':{'op':'fact','key':'a'},
                                     'exception':{'op':'fact','key':'b'}})


def make_transfer_protocol(learning,selection,seeds=(17,29,43),updates=12):
    if (not learning or not selection or type(updates) is not int or updates<1
            or not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int for s in seeds)):
        raise ValueError('nonempty populations and explicit training budget required')
    ids=[r['id'] for r in learning+selection]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate request identity')
    for rows,role in ((learning,'learning'),(selection,'selection')):
        if any(r['split']!=role or not r['supervised']
               or any(v is not True for v in r['supervised'])
               or len(r['supervised'])!=len(r['targets'])
               or any(type(v) is not bool for v in r['targets']) for r in rows):
            raise ValueError('reviewed populations must preserve original roles')
        if role=='selection' and any(r.get('optimization_allowed') is not False for r in rows):
            raise ValueError('selection optimization must be forbidden')
    for key in ('source_group','audio_sha256','recording_id'):
        a={r['provenance'][key] for r in learning};b={r['provenance'][key] for r in selection}
        if a&b:raise ValueError('source separation failed: '+key)
    groups={r['id']:semantic_rule_group(r['provenance']['rule']) for r in learning+selection}
    # Verify stored gold using the actual AST and all supplied dependency facts.
    for row in learning+selection:
        p=row['provenance'];keys=p['dependency_keys'];values=p['dependency_targets']
        if len(keys)!=2 or len(set(keys))!=2 or len(values)!=2 or any(type(v) is not bool for v in values):
            raise ValueError('two reviewed dependency targets required')
        if dependencies(p['rule'])!=set(keys):raise ValueError('dependency identity differs')
        gold=evaluate(p['rule'],{k:k for k in keys},{k:Fact(v,'reviewed') for k,v in zip(keys,values,strict=True)})['value']
        expected={'final_only':[gold],'structured':values+[gold],'reordered':list(reversed(values))+[gold]}
        if row['arm'] not in expected or row['targets']!=expected[row['arm']]:
            raise ValueError('rule or intermediate gold differs')
    train=[r for r in learning if groups[r['id']]!=EXCLUSION_GROUP]
    heldout=[r for r in learning if groups[r['id']]==EXCLUSION_GROUP]
    if not train or not heldout:raise ValueError('both training and held-out semantic groups required')
    schedules={}
    for seed in seeds:
        schedule=balanced_rule_schedule(train,seed)
        by_rule=defaultdict(list)
        for batch in schedule['batches']:by_rule[batch['rule_pair_id']].append(batch)
        if updates % len(by_rule):raise ValueError('equal update count per trained rule pair required')
        rounds=updates//len(by_rule)
        if any(len(batches)<rounds for batches in by_rule.values()):
            raise ValueError('insufficient balanced training batches')
        rng=random.Random(seed);chosen=[]
        for index in range(rounds):
            keys=sorted(by_rule);rng.shuffle(keys)
            chosen.extend(by_rule[key][index] for key in keys)
        schedules[str(seed)]=chosen
    cells={}
    for rows,source in ((learning,'learning_source'),(selection,'reserved_source')):
        for task in ('trained_rule_group','heldout_rule_group'):
            selected=[r for r in rows if (groups[r['id']]==EXCLUSION_GROUP)==(task=='heldout_rule_group')]
            if not selected:raise ValueError('empty source/task evaluation cell')
            # Validate all arms and complements, even for the non-balanced eval population.
            pairs={}
            for r in selected:
                p=r['provenance'];key=(p['recording_id'],p['complement_pair_id'])
                bucket=pairs.setdefault(key,set());item=(r['arm'],p['rule_polarity'])
                if item in bucket:raise ValueError('duplicate matched control')
                bucket.add(item)
            expected=set(itertools.product(('final_only','structured','reordered'),('positive','negative')))
            if any(v!=expected for v in pairs.values()):raise ValueError('incomplete evaluation controls')
            cells[source+':'+task]=dict(request_ids=[r['id'] for r in selected],requests=len(selected),
                recordings=len({r['provenance']['recording_id'] for r in selected}),
                source_groups=len({r['provenance']['source_group'] for r in selected}),
                semantic_groups=sorted({groups[r['id']] for r in selected}))
    return dict(schema='jev-span-rule-transfer-protocol-v1',heldout_semantic_group=EXCLUSION_GROUP,
        symmetry='input renaming + output complement; no input negation',
        training_semantic_groups=sorted({groups[r['id']] for r in train}),
        eligible_training_request_ids=[r['id'] for r in train],schedules=schedules,
        seeds=list(seeds),updates_per_branch=updates,requests_per_update=8,
        evaluation_cells=cells,training_admitted=False,
        scope='Prospective within-SLURP compositional transfer protocol. Requires fresh runs from a fact-only initializer and per-version runtime gates. Not cross-dataset transfer, speaker isolation or evidence of gains.')

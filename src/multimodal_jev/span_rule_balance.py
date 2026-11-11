"""Matched learning controls with complementary rules and truth-table batches.

Each eight-request structured batch contains four distinct source groups, one
per two-fact truth assignment, each under R and NOT R. Every span is balanced
within that mixed-rule batch; individual rules retain their truth-table prior.
"""
from collections import defaultdict
from copy import deepcopy
import itertools
import random



def add_complementary_rules(rows):
    output=[]
    for original in rows:
        if original['split']!='learning':raise ValueError('learning-only rule controls required')
        row=deepcopy(original)
        p=row['provenance']
        if any(type(v) is not bool for v in row['targets']) or not all(row['supervised']):
            raise ValueError('fully supervised Boolean controls required')
        row['provenance']['complement_pair_id']=p['rule_id']
        row['provenance']['rule_polarity']='positive'
        output.append(row)
        inverse=deepcopy(row);p=inverse['provenance']
        p['rule']={'op':'not','arg':p['rule']}
        # Original final question contains the complete rule and definitions.
        # Explicitly negate that rule, preserving the original question verbatim.
        final=inverse['request']['questions'][-1]
        final['schema']['question']=('Decide whether the following complete rule is FALSE. '
            'Return true exactly when the rule is false, and false exactly when it is true. '
            + final['schema']['question'])
        p['rule_polarity']='negative';p['rule_id']=p['rule_id']+':negated'
        p['structural_family']='not('+p['structural_family']+')'
        inverse['id']=inverse['id']+':negated'
        inverse['targets'][-1]=not inverse['targets'][-1]
        output.append(inverse)
    if len({r['id'] for r in output})!=len(output):raise ValueError('duplicate request ID')
    return output


def balanced_rule_schedule(rows,seed):
    if type(seed) is not int:raise ValueError('integer seed required')
    by_case=defaultdict(dict)
    for row in rows:
        p=row['provenance']
        key=(p['complement_pair_id'],p['recording_id'],row['arm'])
        polarity=p['rule_polarity']
        if polarity not in ('positive','negative') or polarity in by_case[key]:
            raise ValueError('duplicate or unknown rule polarity')
        by_case[key][polarity]=row
    units=defaultdict(dict)
    for (rule,recording,arm),pair in sorted(by_case.items()):
        if set(pair)!= {'positive','negative'}:raise ValueError('both rule polarities required')
        a,b=pair['positive'],pair['negative']
        if (a['targets'][:-1]!=b['targets'][:-1] or type(a['targets'][-1]) is not bool
                or b['targets'][-1] is not (not a['targets'][-1])
                or a['provenance']['dependency_targets']!=b['provenance']['dependency_targets']):
            raise ValueError('complementary targets differ')
        units[(rule,recording)][arm]=pair
    buckets=defaultdict(lambda:defaultdict(list))
    for (rule,recording),arms in sorted(units.items()):
        if set(arms)!= {'final_only','structured','reordered'}:
            raise ValueError('three matched arms required')
        row=arms['structured']['positive'];p=row['provenance']
        signature=tuple(p['dependency_targets'])
        if len(signature)!=2 or any(type(v) is not bool for v in signature):
            raise ValueError('two Boolean facts required')
        if row['targets'][:2]!=list(signature):raise ValueError('structured fact labels differ')
        buckets[rule][signature].append((recording,p['source_group'],arms))
    rng=random.Random(seed);batches=[];excluded=[]
    signatures=list(itertools.product((False,True),repeat=2))
    for rule,strata in sorted(buckets.items()):
        if set(strata)!=set(signatures):raise ValueError('all four reviewed truth assignments required')
        # A source appearing under distinct signatures is a data contradiction.
        sources=[x[1] for entries in strata.values() for x in entries]
        if len(set(sources))!=len(sources):raise ValueError('distinct source groups required within rule')
        for entries in strata.values():rng.shuffle(entries)
        count=min(map(len,strata.values()))
        for signature,entries in strata.items():
            excluded.extend(dict(rule_id=rule,recording_id=x[0],reason='truth_stratum_downsample') for x in entries[count:])
        for index in range(count):
            selected=[strata[s][index] for s in signatures]
            order=[(i,p) for i in range(4) for p in ('positive','negative')];rng.shuffle(order)
            batch=dict(rule_pair_id=rule,recording_ids=[x[0] for x in selected],source_groups=[x[1] for x in selected],
                arms={arm:[selected[i][2][arm][p]['id'] for i,p in order]
                      for arm in ('final_only','structured','reordered')})
            batches.append(batch)
    rng.shuffle(batches)
    return dict(schema='jev-span-complementary-rule-schedule-v1',seed=seed,batches=batches,excluded=excluded,
        recordings_per_batch=4,requests_per_arm_batch=8,
        scope='Exact per-span balance across paired R/NOT R; individual-rule priors unchanged; learning-only, no independent sample expansion')

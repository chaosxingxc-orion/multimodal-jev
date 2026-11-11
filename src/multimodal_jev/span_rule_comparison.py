"""Paired finite-decision gains with explicit training exposure categories."""
from collections import defaultdict
from copy import deepcopy


def compare_predictions(rows, before, after, training_rows):
    """Do not call new requests on familiar audio held-out generalization.

    Baseline initialization may have seen every source; categories here only
    describe exposure to the new rule update. Counterfactual rule polarities
    remain grouped, and no request-level independence or significance is assumed.
    """
    expected={r['id']:r for r in rows}
    a={r['id']:r for r in before}; b={r['id']:r for r in after}
    if (len(expected)!=len(rows) or len(a)!=len(before) or len(b)!=len(after)
            or expected.keys()!=a.keys() or a.keys()!=b.keys() or not rows or not training_rows):
        raise ValueError('complete unique predictions and training exposure required')
    trained_sources={r['provenance']['source_group'] for r in training_rows}
    trained_rules={r['provenance']['complement_pair_id'] for r in training_rows}
    groups=defaultdict(list)
    for uid,row in expected.items():
        old,new=a[uid],b[uid]
        x,y=deepcopy(old['request_identity']),deepcopy(new['request_identity'])
        x.pop('policy_id'); y.pop('policy_id')
        if x!=y or x['request']!=row['request']:
            raise ValueError('input changed across compared policies')
        if any(len(r['predictions'])!=len(row['targets']) or
               any(type(v) is not bool for v in r['predictions']) for r in (old,new)):
            raise ValueError('Boolean decision outputs required')
        source_seen=row['provenance']['source_group'] in trained_sources
        rule_seen=row['provenance']['complement_pair_id'] in trained_rules
        item=dict(old=old['predictions'][-1]==row['targets'][-1],
                  new=new['predictions'][-1]==row['targets'][-1],
                  gold=row['targets'][-1],rule=row['provenance']['rule_id'],
                  source=row['provenance']['source_group'])
        for category in ('all',f'update_source_{"seen" if source_seen else "unseen"}_rule_{"seen" if rule_seen else "unseen"}'):
            groups[(row['arm'],category)].append(item)
    results=[]
    for (arm,category),items in sorted(groups.items()):
        rule_groups=defaultdict(list)
        for item in items:rule_groups[item['rule']].append(item)
        balanced={}; covered=0
        for version in ('old','new'):
            scores=[]
            for group in rule_groups.values():
                strata=[[r[version] for r in group if r['gold'] is value] for value in (False,True)]
                if all(strata):scores.append(sum(sum(s)/len(s) for s in strata)/2)
            balanced[version]=sum(scores)/len(scores) if scores else None
            covered=len(scores)
        wins=sum(r['new'] and not r['old'] for r in items)
        losses=sum(r['old'] and not r['new'] for r in items)
        results.append(dict(arm=arm,category=category,n=len(items),sources=len({r['source'] for r in items}),
            before_accuracy=sum(r['old'] for r in items)/len(items),
            after_accuracy=sum(r['new'] for r in items)/len(items),
            delta_accuracy=(wins-losses)/len(items),wins=wins,losses=losses,ties=len(items)-wins-losses,
            before_macro_rule_ba=balanced['old'],after_macro_rule_ba=balanced['new'],
            rules_with_both_labels=covered,rules_total=len(rule_groups)))
    return dict(groups=results,
        scope='Exposure relative only to this rule update; initializer previously saw learning sources. No held-out/generalization/significance claim.')

"""Learning-only derived rule controls from authenticated reviewed audio facts.

This does not create an evaluation split or new acoustic evidence. Its labels
are compositions of reviewed labels, never the absence of an annotation.
"""
from collections import Counter
from dataclasses import asdict
import itertools
import json

from .decision_rules import Fact, evaluate, family, question
from .span_contract import SpanQuestion, SpanRequest
from .task_schema import DecisionSchema


def build_rule_controls(bundles):
    rows=[];excluded=[];counts=Counter()
    for bundle in bundles:
        original=bundle['request']
        if len(original.questions)!=3 or len(bundle['targets'])!=3 or len(bundle['supervised'])!=3:
            raise ValueError('three reviewed Boolean facts required')
        if any(q.schema.decision_type!='noul' or q.schema.allow_unknown for q in original.questions):
            raise ValueError('binary reviewed fact definitions required')
        keys=[q.id for q in original.questions]
        definitions={q.id:q.schema.question for q in original.questions}
        facts={key:Fact(value if active else None,'reviewed' if active else 'unavailable')
               for key,value,active in zip(keys,bundle['targets'],bundle['supervised'],strict=True)}
        for ia,ib in itertools.combinations(range(3),2):
            a,b=keys[ia],keys[ib]
            A={'op':'fact','key':a};B={'op':'fact','key':b}
            rules=[('both',{'op':'all','args':[A,B]}),('either',{'op':'any','args':[A,B]}),
                   ('first_without_second',{'op':'except','base':A,'exception':B})]
            for rule_name,rule in rules:
                rule_id=f'{a}:{b}:{rule_name}'
                # All arms need both intermediate labels for matched supervision.
                # Do not silently admit a rule via short-circuit with missing gold.
                if not all(bundle['supervised'][i] for i in (ia,ib)):
                    excluded.append(dict(id=bundle['id'],rule_id=rule_id,reason='unreviewed_dependency'))
                    continue
                value=evaluate(rule,definitions,facts)['value']
                if type(value) is not bool:raise ValueError('fully reviewed rule must resolve')
                final=SpanQuestion('final',DecisionSchema('configured-rule','noul',question(rule,definitions),allow_unknown=False))
                requirements=original.requirements+('Apply the explicitly supplied Boolean rule; do not substitute a habitual task rule.',)
                def intermediate(index,identifier):
                    return SpanQuestion(identifier,original.questions[index].schema)
                arms=[('final_only',(final,),(value,)),
                      ('structured',(intermediate(ia,'q1'),intermediate(ib,'q2'),final),
                       (bundle['targets'][ia],bundle['targets'][ib],value)),
                      ('reordered',(intermediate(ib,'q1'),intermediate(ia,'q2'),final),
                       (bundle['targets'][ib],bundle['targets'][ia],value))]
                for arm,questions,targets in arms:
                    request=SpanRequest(original.audio,questions,'final',requirements)
                    rows.append(dict(id=f'{bundle["id"]}::{rule_id}::{arm}',split='learning',arm=arm,
                        request=request.model_input(),targets=list(targets),supervised=[True]*len(targets),
                        provenance=dict(recording_id=bundle['id'],source_group=bundle['source_group'],
                            audio_sha256=bundle['audio_sha256'],decision_ids=bundle['decision_ids'],
                            rule_id=rule_id,rule=rule,structural_family=family(rule,definitions),
                            dependency_keys=[a,b],dependency_targets=[bundle['targets'][ia],bundle['targets'][ib]],
                            review_scope='Inherited reference-semantics review; not a new listening review')))
                    counts[(rule_id,arm,value)]+=1
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate derived request identity')
    return rows,dict(recordings=len(bundles),source_groups=len({b['source_group'] for b in bundles}),
        derived_rows=len(rows),matched_triplets=len(rows)//3,excluded=excluded,
        counts=[dict(rule_id=r,arm=a,label=v,count=n) for (r,a,v),n in sorted(counts.items())],
        scope='Learning-only derived Boolean controls; no new independent recordings, held-out split, task gains or CoT evidence')


def predicted_fact_rule(rule, predictions):
    """Deployed deterministic executor baseline; takes predictions, never gold."""
    definitions={k:k for k in predictions}
    if any(type(v) is not bool for v in predictions.values()):
        raise ValueError('Boolean model predictions required')
    result=evaluate(rule,definitions,{k:Fact(v,'reviewed') for k,v in predictions.items()})
    if type(result['value']) is not bool:raise ValueError('complete predicted dependencies required')
    return result['value']

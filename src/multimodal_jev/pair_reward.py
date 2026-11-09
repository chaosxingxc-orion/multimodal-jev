"""Correctness-gated paired reward for the preregistered pair-GRPO experiment.

The caller must sample complete groups before computing these rewards. Applying
this function independently to shuffled minibatches changes the objective and
is unsupported. This module is not yet wired into a verl reward manager.
"""
from __future__ import annotations
from collections import defaultdict
from .reward import score_rollout


PAIRS=(('native','permutation','invariance'),
       ('native','rebind','definition_discrimination'),
       ('native','missing_candidate','candidate_coverage'),
       ('noul.True','noul.False','binary_discrimination'))


def semantic_description(example, value):
    return next((c.description for c in example.schema.candidates if c.id==value),None)


def validate_relation(left,right,relation):
    if left.audio!=right.audio or left.split!=right.split:
        raise ValueError('paired examples must share audio and split')
    if not left.metadata.get('audio_sha256') or left.metadata.get('audio_sha256')!=right.metadata.get('audio_sha256'):
        raise ValueError('paired audio hash mismatch')
    if relation in {'invariance','definition_discrimination'}:
        if left.schema.decision_type!='choice' or right.schema.decision_type!='choice':
            raise ValueError('pair must contain choices')
        if semantic_description(left,left.target)!=semantic_description(right,right.target):
            raise ValueError('pair target meanings differ')
        if relation=='invariance':
            if left.target!=right.target or set(left.schema.candidates)!=set(right.schema.candidates):
                raise ValueError('permutation changed candidate definitions')
        else:
            if left.target==right.target:
                raise ValueError('rebind did not change correct candidate identity')
            if {c.id for c in left.schema.candidates}!={c.id for c in right.schema.candidates}:
                raise ValueError('rebind changed candidate ID set')
            if sorted(c.description for c in left.schema.candidates)!=sorted(c.description for c in right.schema.candidates):
                raise ValueError('rebind changed candidate definition set')
    elif relation=='candidate_coverage':
        remaining={c.id:c.description for c in right.schema.candidates}
        expected={c.id:c.description for c in left.schema.candidates if c.id!=left.target}
        if right.target is not None or remaining!=expected:
            raise ValueError('missing-candidate pair does not remove exactly the answer')
    else:
        if left.schema.decision_type!='noul' or right.schema.decision_type!='noul':
            raise ValueError('binary pair must contain noul decisions')
        if left.target is not True or right.target is not False or left.schema.question==right.schema.question:
            raise ValueError('binary pair must have different questions and opposite truth labels')


def paired_rewards(examples,predictions,bonus_weight=0.2):
    """Return rewards in input order; predictions are decoded JSON values.

    Each repeated rollout sample must have a distinct caller-managed batch;
    duplicate group/variant rows are rejected instead of silently overwritten.
    Every participating task gets the mean bonus over its declared pair edges.
    """
    if len(examples)!=len(predictions) or not examples:
        raise ValueError('nonempty aligned examples/predictions required')
    if not 0<=bonus_weight<=1:
        raise ValueError('bonus weight must be in [0,1]')
    groups=defaultdict(dict)
    scores=[score_rollout(ex,value) for ex,value in zip(examples,predictions)]
    bonus=[[] for _ in examples]
    relation_stats=defaultdict(lambda:{'pairs':0,'both_correct':0})
    for index,ex in enumerate(examples):
        group=ex.metadata.get('source_group');variant=ex.metadata.get('variant')
        if not group or not variant:
            raise ValueError('source_group and variant are required')
        if variant in groups[group]:
            raise ValueError('duplicate group/variant; pair rollout replicate indices explicitly')
        groups[group][variant]=index
    for group,views in groups.items():
        expected={name for pair in PAIRS for name in pair[:2]}
        if set(views)!=expected:
            raise ValueError(f'incomplete or unexpected pair group: {group}')
        for left,right,relation in PAIRS:
            i,j=views[left],views[right]
            validate_relation(examples[i],examples[j],relation)
            correct=float(bool(scores[i]['semantic']) and bool(scores[j]['semantic']))
            bonus[i].append(correct);bonus[j].append(correct)
            relation_stats[relation]['pairs']+=1
            relation_stats[relation]['both_correct']+=int(correct)
    result=[]
    for base,edges in zip(scores,bonus):
        paired=sum(edges)/len(edges)
        result.append({**base,'pair_bonus':paired,'total':base['total']+bonus_weight*paired})
    return result,dict(relation_stats)

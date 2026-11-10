"""Learning-only SFC rewards using the unchanged strict benchmark semantics.

This is an offline reward component, not a verl rollout/actor integration.
Call once for each aligned rollout replicate after gathering the entire bundle.
"""
import math

from .sfc_jev_score import score_bundle


def bundle_rewards(request,gold,predictions,*,learning_groups,mode='independent',bonus_weight=0.2):
    """Strict decision rewards plus optional correctness-gated function bonus.

    `learning_groups` must come from the registered/reproduced TRAIN partition.
    A caller cannot pass selection/test rows or silently omit a function/decision.
    Outputs follow input decision order; the caller owns replicate alignment and
    same-prompt GRPO grouping. No bonus is given for mere logical consistency.
    """
    if (type(learning_groups) not in (set,frozenset) or not learning_groups
            or any(type(g) is not str or not g for g in learning_groups)
            or request['split']!='train_id' or request['source_group'] not in learning_groups):
        raise ValueError('registered TRAIN learning group required')
    if mode not in ('independent','function-conjunction'):
        raise ValueError('registered SFC reward mode required')
    if (type(bonus_weight) not in (int,float) or not math.isfinite(bonus_weight)
            or not 0<=bonus_weight<=1):
        raise ValueError('finite bonus weight in [0,1] required')
    report=score_bundle(request,gold,predictions)
    scored=report['decisions'];groups={}
    for decision in request['input']['decisions']:
        prefix,suffix=decision['id'].rsplit('.',1)
        groups.setdefault(prefix,{})[suffix]=decision['id']
    if not groups or any(set(g)!={'present','absent','calls'} for g in groups.values()):
        raise ValueError('complete original function triples required')
    conjunction={prefix:all(scored[key]['correct'] for key in keys.values()) for prefix,keys in groups.items()}
    records=[]
    for decision in request['input']['decisions']:
        key=decision['id'];result=scored[key];prefix=key.rsplit('.',1)[0]
        semantic=float(result['correct'])
        # Separate penalties avoid double-charging invalid JSON as schema failure.
        format_penalty=-1.0 if not result['format_valid'] else 0.0
        schema_penalty=-0.1 if result['format_valid'] and not result['schema_valid'] else 0.0
        base=semantic+format_penalty+schema_penalty
        bonus=bonus_weight*float(conjunction[prefix]) if mode=='function-conjunction' else 0.0
        records.append(dict(decision_id=key,semantic=semantic,format_penalty=format_penalty,
            schema_penalty=schema_penalty,base_reward=base,function_bonus=bonus,total=base+bonus,
            format_valid=result['format_valid'],schema_valid=result['schema_valid']))
    return dict(schema='sfc-jev-learning-reward-v1',mode=mode,bonus_weight=bonus_weight,
        example_id=request['id'],source_group=request['source_group'],decisions=records,
        function_triples=len(groups),correct_function_triples=sum(conjunction.values()),
        # Equal decision weighting here equals equal function weighting because
        # the input-only protocol declares exactly three decisions per function.
        bundle_mean_reward=math.fsum(r['total'] for r in records)/len(records),
        native_calls_exact=report['native_calls_exact'],all_decisions_correct=report['all_decisions_correct'],
        consistent_function_triples=report['counts'].get('consistent_function_triples',0),
        scope='One complete learning bundle/replicate. Strict raw scorer reused; no output repair, '
              'Noul gating, execution-success reward, confidence bonus, rollout or optimizer update. '
              'Caller must reproduce partition and align all sampling replicates before GRPO grouping.')

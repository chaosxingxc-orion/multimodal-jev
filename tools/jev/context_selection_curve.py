"""Summarize predeclared selection checkpoints, preserving incomplete curves."""
import math


def summarize(points, experiment, *, initial_policy_id):
    evaluation=experiment['evaluation'];planned=evaluation['checkpoints']
    if (not planned or planned[0]!=0 or planned!=sorted(set(planned)) or
            planned[-1]!=experiment['task_stage']['updates']):
        raise ValueError('ordered predeclared task checkpoints required')
    steps=[p['step'] for p in points]
    if not steps or steps!=planned[:len(steps)]:
        raise ValueError('complete ordered evaluation prefix including step0 required')
    if points[0]['result']['policy_id']!=initial_policy_id:
        raise ValueError('step0 must be the fixed task-stage initialization')
    ids=points[0]['result']['metrics']['candidate_ids']
    if len(ids)!=60 or len(set(ids))!=60:raise ValueError('fixed 60 candidate ontology required')
    fields=[evaluation['primary'],*evaluation['secondary']]
    curve=[]
    for point in points:
        result=point['result'];metrics=result['metrics']
        if (result['role']!='selection' or result['release_sha256']!=experiment['release_sha256'] or
                metrics['queries']!=experiment['selection_recordings'] or
                metrics['threshold']!=evaluation['set_threshold'] or metrics['candidate_ids']!=ids):
            raise ValueError('fixed selection population, candidate order and threshold required')
        values={k:metrics[k] for k in fields}
        for key,value in values.items():
            if key=='candidate_binary_nll' and metrics['binary_nll_infinite']:
                if value is not None:raise ValueError('infinite NLL must remain explicit null')
                continue
            if key=='set_precision' and value is None and metrics['set_precision_defined_queries']==0:
                continue
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                raise ValueError('invalid metric: '+key)
            limit=60 if key=='mean_selected_count' else (math.inf if key=='candidate_binary_nll' else 1)
            if value>limit:raise ValueError('metric out of range: '+key)
        curve.append(dict(step=point['step'],policy_id=result['policy_id'],
            predictions_sha256=result['predictions_sha256'],metrics=values,
            binary_nll_infinite=metrics['binary_nll_infinite'],
            set_precision_defined_queries=metrics['set_precision_defined_queries']))
    primary=evaluation['primary'];base=curve[0]['metrics']
    for row in curve:
        row['delta_from_step0']={k:(None if row['metrics'][k] is None or base[k] is None
                                      else row['metrics'][k]-base[k]) for k in fields}
    falls=[dict(previous=a['step'],step=b['step'],delta=b['metrics'][primary]-a['metrics'][primary])
           for a,b in zip(curve,curve[1:]) if b['metrics'][primary]<a['metrics'][primary]]
    complete=steps==planned
    chosen=None
    if complete:
        chosen=min(curve,key=lambda x:(-x['metrics'][primary],
            math.inf if x['binary_nll_infinite'] else x['metrics']['candidate_binary_nll'],x['step']))
    return dict(complete=complete,planned_steps=planned,observed_steps=steps,
        missing_steps=planned[len(steps):],initial_policy_id=initial_policy_id,
        selected_checkpoint=None if chosen is None else dict(step=chosen['step'],policy_id=chosen['policy_id']),
        curve=curve,primary_adjacent_decreases=falls,
        last_primary_delta=curve[-1]['metrics'][primary]-base[primary],
        scope='Descriptive fixed-selection curve; incomplete prefixes never select a checkpoint. '
              'No significance, RL-only causal attribution, test performance or transfer claim.')

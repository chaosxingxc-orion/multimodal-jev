"""Matched direct/structured intent tasks, with labels outside model inputs.

Scenario/action labels are deterministic projections of an admitted joint intent,
not independently annotated reasoning steps or evidence of multi-hop reasoning.
"""
from .span_contract import SpanQuestion,SpanRequest
from .task_schema import Candidate,DecisionSchema
from .slurp_training import validate_admission


def compile_intent(audio,task,arm):
    if arm not in ('D','Q','P'):raise ValueError('fixed D/Q/P arm required')
    if task['kind']!='intent':raise ValueError('intent task required')
    schema=DecisionSchema.from_dict(task['schema']);pairs=task['candidate_pairs']
    if (schema.decision_type!='choice' or schema.allow_unknown
            or set(pairs)!={c.id for c in schema.candidates}
            or any(type(p)!=list or len(p)!=2 or any(type(x)!=str or not x for x in p) for p in pairs.values())
            or len({tuple(p) for p in pairs.values()})!=len(pairs)):
        raise ValueError('unique complete canonical pair mapping required')
    # Same final candidate knowledge and final question for every arm.
    final=SpanQuestion('intent',schema);questions=[]
    if arm!='D':
        for i,name in enumerate(('scenario','action')):
            values=sorted({p[i] for p in pairs.values()})
            questions.append(SpanQuestion(name,DecisionSchema(
                id='slurp-intent-'+name+'-v1',decision_type='choice',
                question=('Which listed scenario describes the spoken request?' if i==0 else
                          'Which listed action describes the spoken request?'),
                candidates=tuple(Candidate(v,('Scenario: ' if i==0 else 'Action: ')+v+'.') for v in values),
                allow_unknown=False)))
    questions.append(final)
    return SpanRequest(audio,tuple(questions),'intent',tuple(task['thinking']['requirements']))


def training_targets(example,task,arm,process_weight=.2):
    """Admitted joint intent projects to known components; no text enters input."""
    import math
    validate_admission(example)
    if type(process_weight) not in (int,float) or not math.isfinite(process_weight) or not 0<=process_weight<1:
        raise ValueError('bounded process reward budget required')
    from .slurp_training import content_hash
    if (example.metadata['slurp_review']['task_sha256']!=content_hash(task)
            or example.schema!=DecisionSchema.from_dict(task['schema'])):
        raise ValueError('task must match admitted supervision')
    request=compile_intent(example.audio,task,arm)
    if example.target not in task['candidate_pairs']:raise ValueError('unknown joint target')
    pair=task['candidate_pairs'][example.target]
    values={'scenario':pair[0],'action':pair[1],'intent':example.target}
    return request,dict(targets=[q.values.index(values[q.id]) for q in request.questions],
        supervised=[True]*len(request.questions),
        span_weights=[1. if q.id=='intent' else process_weight/2 if arm=='P' else 0. for q in request.questions],
        target_values=[values[q.id] for q in request.questions],
        process_supervision='deterministic canonical pair projection of reviewed joint intent; no independent process annotation',
        source_group=example.metadata['source_group'],decision_sha256=example.metadata['slurp_review']['decision_sha256'])

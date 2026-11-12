"""Label-free FSC direct/structured Span templates and semantic rebinding.

Inference depends only on audio and a frozen complete task configuration.
Targets are consumed separately by score_one, never by compile_request.
"""
import hashlib
from .span_contract import SpanQuestion,SpanRequest
from .task_schema import Candidate,DecisionSchema

FIELDS=('action','object','location')


def validate(config):
    if config['schema']!='fsc-span-transfer-v1' or config['input_modality']!='audio':
        raise ValueError('fixed audio transfer configuration required')
    commands=config['commands']
    if len(commands)!=31 or len({c['id'] for c in commands})!=31:
        raise ValueError('complete 31-command support required')
    tuples=[]
    for command in commands:
        if not command['description'].strip() or not command['id']:raise ValueError('described command required')
        values=tuple(command['values'][f] for f in FIELDS);tuples.append(values)
        for field,value in zip(FIELDS,values,strict=True):
            if value not in config['components'][field] or not config['components'][field][value].strip():
                raise ValueError('described component value required')
    if len(set(tuples))!=31:raise ValueError('distinct legal commands required')
    for i,field in enumerate(FIELDS):
        if set(config['components'][field])!={t[i] for t in tuples}:
            raise ValueError('component support must equal legal command projection')
    if [len(config['components'][f]) for f in FIELDS]!=[6,14,4]:
        raise ValueError('native FSC component support differs')
    if not config['thinking_requirements']:raise ValueError('configured thinking requirements required')


def compile_request(audio,config,template='direct',*,variant='native',seed=0):
    validate(config)
    if template not in ('direct','structured') or variant not in ('native','permutation','rebind') or type(seed)!=int:
        raise ValueError('fixed template and input-independent binding required')
    def candidates(entries,field):
        rows=list(entries)
        if variant=='permutation':
            rows.sort(key=lambda x:hashlib.sha256(f'{seed}:{field}:{x[0]}'.encode()).hexdigest())
        if variant=='rebind':
            # Shift descriptions across positions as well as opaque IDs. Native
            # symbols A/B/... otherwise would still name the same semantics.
            shift=1+seed%(len(rows)-1);rows=rows[shift:]+rows[:shift]
            return tuple(Candidate(f'{field}_bound_{i:02d}',text) for i,(_,text,_) in enumerate(rows)),{
                f'{field}_bound_{i:02d}':value for i,(_,_,value) in enumerate(rows)}
        return tuple(Candidate(key,text) for key,text,_ in rows),{key:value for key,_,value in rows}
    questions=[];bindings={}
    if template=='structured':
        for field in FIELDS:
            entries=[(v,d,v) for v,d in config['components'][field].items()]
            options,bindings[field]=candidates(entries,field)
            questions.append(SpanQuestion(field,DecisionSchema('fsc-span-'+field+'-v1','choice',
                config['questions'][field],options,allow_unknown=False)))
    entries=[(c['id'],c['description'],c['values']) for c in config['commands']]
    options,bindings['final']=candidates(entries,'final')
    questions.append(SpanQuestion('final',DecisionSchema('fsc-span-final-v1','choice',
        config['questions']['final'],options,allow_unknown=False)))
    return SpanRequest(str(audio),tuple(questions),'final',tuple(config['thinking_requirements'])),bindings


def score_one(prediction,bindings,gold):
    """Score model Final separately from component agreement; no answer repair."""
    if set(gold)!=set(FIELDS) or set(prediction)!=set(bindings):raise ValueError('complete configured output and native gold required')
    if any(value not in bindings[field] for field,value in prediction.items()):raise ValueError('output outside configured support')
    if not any(value==gold for value in bindings['final'].values()):raise ValueError('gold outside legal commands')
    semantic={field:bindings[field][value] for field,value in prediction.items()}
    final=semantic['final'];structured=set(bindings)==set(FIELDS)|{'final'}
    return dict(final_correct=final==gold,final_component_correct={f:final[f]==gold[f] for f in FIELDS},
        process_correct={f:semantic[f]==gold[f] for f in FIELDS} if structured else None,
        compatible=all(semantic[f]==final[f] for f in FIELDS) if structured else None,
        semantic_prediction=semantic)

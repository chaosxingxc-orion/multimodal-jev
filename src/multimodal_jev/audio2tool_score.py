"""Strict ordered JSON metrics alongside explicitly named official primitives.

Gold calls must come from a separately authenticated target conversion. This
module never opens annotations or supplies any actor inputs. Runtime failures
and invalid JSON remain in the declared record denominator.
"""
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

OFFICIAL_METRICS_SHA256='b092cc5ba1bac59d58dbfeb56c7f7cf63f1970df59892ea7b56ee24ac2c86802'


def typed(value):
    """Type-aware equality, including nested values; object order irrelevant."""
    if value is None:return ('null',)
    if type(value) in (str,bool,int):return (type(value).__name__,value)
    if type(value) is float and math.isfinite(value):return ('float',value)
    if type(value) is list:return ('list',tuple(typed(x) for x in value))
    if type(value) is dict and all(type(k) is str for k in value):
        return ('object',tuple((k,typed(v)) for k,v in sorted(value.items())))
    raise ValueError('finite JSON values required')


def validate_calls(calls):
    if type(calls) is not list:raise ValueError('JSON call array required')
    for call in calls:
        if (type(call) is not dict or set(call)!={'name','arguments'}
                or type(call['name']) is not str or not call['name']
                or type(call['arguments']) is not dict):
            raise ValueError('name and arguments call object required')
        typed(call)
    return calls


def parse_prediction(raw):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('duplicate JSON key')
            result[k]=v
        return result
    def constant(_):raise ValueError('nonfinite JSON constant')
    if type(raw) is not str:raise ValueError('raw model text required')
    try:return validate_calls(json.loads(raw,object_pairs_hook=pairs,parse_constant=constant))
    except (json.JSONDecodeError,RecursionError) as exc:raise ValueError('invalid JSON response') from exc


def load_official(path):
    """Load the audited upstream metrics file; no network or source replacement."""
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=OFFICIAL_METRICS_SHA256:
        raise ValueError('pinned official metric source required')
    name='_audio2tool_pinned_metrics'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    try:spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name,None);raise
    return module


def slots(calls):
    # Index binding deliberately penalizes reorderings and values transferred
    # between repeated calls. This is a project metric, not paper micro-F1.
    return Counter((i,c['name'],k,typed(v)) for i,c in enumerate(calls) for k,v in c['arguments'].items())


def score_record(raw,gold,official,*,error=None):
    validate_calls(gold)
    if error is not None and type(error) is not str:raise ValueError('error must be string or null')
    valid=False;pred=[]
    if error is None:
        try:pred=parse_prediction(raw);valid=True
        except ValueError:pass
    p,g=slots(pred),slots(gold)
    tp=sum((p&g).values());fp=sum((p-g).values());fn=sum((g-p).values())
    primary_pred=pred[0] if pred else dict(name='',arguments={})
    primary_gold=gold[0] if gold else dict(name='',arguments={})
    parameters=official.compute_parameter_accuracy(primary_pred['arguments'],primary_gold['arguments'])
    multi=official.compute_multi_tool_metrics([c['name'] for c in pred],[c['name'] for c in gold])
    return dict(format_valid=valid,runtime_error=error is not None,
        strict_ordered_exact=valid and typed(pred)==typed(gold),
        strict_ordered_tools_exact=valid and [c['name'] for c in pred]==[c['name'] for c in gold],
        indexed_slot_tp=tp,indexed_slot_fp=fp,indexed_slot_fn=fn,
        official_primary_exact=valid and official.compute_exact_match(primary_pred['name'],primary_pred['arguments'],
                                                                       primary_gold['name'],primary_gold['arguments']),
        official_primary_parameter_f1=parameters['f1'] if valid else 0.,
        official_tool_set_exact=valid and multi['all_tools_match'])


def evaluate(targets,predictions,official):
    """Require one terminal prediction per target; no silent dropped records.

    Missing infrastructure results are an incomplete evaluation, not fabricated
    failures. Explicit runtime-error rows remain counted as incorrect.
    """
    expected={r['id']:r for r in targets};observed={r['id']:r for r in predictions}
    if (not targets or len(expected)!=len(targets) or len(observed)!=len(predictions)
            or set(expected)!=set(observed)):
        raise ValueError('complete unique target and terminal-prediction coverage required')
    details=[]
    for key,target in expected.items():
        pred=observed[key]
        if set(pred)!={'id','raw_text','error'}:raise ValueError('explicit raw text and error required')
        details.append(dict(id=key,**score_record(pred['raw_text'],target['calls'],official,error=pred['error'])))
    n=len(details)
    report=dict(records=n,format_valid=sum(r['format_valid'] for r in details),
                runtime_errors=sum(r['runtime_error'] for r in details))
    for key in ('strict_ordered_exact','strict_ordered_tools_exact','official_primary_exact',
                'official_primary_parameter_f1','official_tool_set_exact'):
        report[key]=sum(r[key] for r in details)/n
    tp,fp,fn=(sum(r['indexed_slot_'+k] for r in details) for k in ('tp','fp','fn'))
    report.update(indexed_slot_tp=tp,indexed_slot_fp=fp,indexed_slot_fn=fn,
                  indexed_slot_micro_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                  scope='All declared records including errors; official primitive metrics under local JSON parsing. Not unchanged official evaluator or paper protocol.')
    return report,details


def validate_projected_target(target,official):
    """Check coverage declarations rather than trusting an unknown-to-empty repair."""
    from .audio2tool_targets import literal_call
    primary=target['primary'];validate_calls([primary])
    annotations=target['additional_annotations'];expected=[primary['name']];complete=[primary]
    missing=False
    for a in annotations:
        if set(a)!={'raw','name','arguments','argument_evidence'}:
            raise ValueError('explicit extra-call evidence required')
        if official.normalize_tool_name(a['raw'])!=official.normalize_tool_name(a['name']):
            raise ValueError('extra tool label differs')
        expected.append(a['raw'])
        if a['argument_evidence']=='unavailable':
            if a['arguments'] is not None:raise ValueError('unknown arguments must remain None')
            missing=True
        elif a['argument_evidence']=='explicit_literal_call':
            call=dict(name=a['name'],arguments=a['arguments']);validate_calls([call])
            if typed(literal_call(a['raw']))!=typed(call):raise ValueError('explicit argument evidence differs')
            complete.append(call)
        else:raise ValueError('unknown argument evidence status')
    if (target['official_expected_tools']!=expected
            or target['full_call_argument_coverage']!=('incomplete' if missing else 'complete')
            or typed(target['full_calls'])!=typed(None if missing else complete)):
        raise ValueError('target coverage or call evidence differs')


def evaluate_projected(targets,predictions,official):
    """Full-record primary/name metrics; full-argument subset has its own denominator."""
    expected={r['id']:r for r in targets};observed={r['id']:r for r in predictions}
    if (not targets or len(expected)!=len(targets) or len(observed)!=len(predictions)
            or set(expected)!=set(observed)):
        raise ValueError('complete unique target and terminal-prediction coverage required')
    details=[]
    for key,target in expected.items():
        validate_projected_target(target,official)
        record=observed[key]
        if set(record)!={'id','raw_text','error'} or (record['error'] is not None and type(record['error']) is not str):
            raise ValueError('explicit terminal raw text and error required')
        pred=[];valid=False
        if record['error'] is None:
            try:pred=parse_prediction(record['raw_text']);valid=True
            except ValueError:pass
        primary=target['primary'];first=pred[0] if pred else dict(name='',arguments={})
        param=official.compute_parameter_accuracy(first['arguments'],primary['arguments'])
        names=[c['name'] for c in pred]
        tools=official.compute_multi_tool_metrics(names,target['official_expected_tools'])
        full=target['full_calls']
        detail=dict(id=key,tier=target['tier'],format_valid=valid,runtime_error=record['error'] is not None,
            strict_primary_exact=bool(valid and pred and typed(first)==typed(primary)),
            official_primary_exact=bool(valid and pred and official.compute_exact_match(
                first['name'],first['arguments'],primary['name'],primary['arguments'])),
            official_primary_tool_accuracy=bool(valid and pred and official.compute_tool_accuracy(first['name'],primary['name'])),
            official_primary_parameter_f1=param['f1'] if valid else 0.,
            official_tool_set_exact=bool(valid and tools['all_tools_match']),
            official_tool_set_f1=tools['tool_f1'] if valid else 0.,
            strict_ordered_names_exact=bool(valid and names==[primary['name']]+[a['name'] for a in target['additional_annotations']]),
            full_arguments_available=full is not None,
            strict_full_calls_exact=None if full is None else bool(valid and typed(pred)==typed(full)))
        details.append(detail)
    metrics=('strict_primary_exact','official_primary_exact','official_primary_tool_accuracy',
             'official_primary_parameter_f1','official_tool_set_exact','official_tool_set_f1','strict_ordered_names_exact')
    def summarize(items):
        n=len(items);covered=[r for r in items if r['full_arguments_available']]
        return dict(records=n,format_valid=sum(r['format_valid'] for r in items),
            runtime_errors=sum(r['runtime_error'] for r in items),
            full_record_metrics={k:sum(r[k] for r in items)/n for k in metrics},
            full_arguments_subset=dict(eligible_records=len(covered),unavailable_records=n-len(covered),
                strict_full_calls_exact=sum(r['strict_full_calls_exact'] for r in covered)/len(covered) if covered else None),
            full_population_strict_calls_exact=(sum(r['strict_full_calls_exact'] for r in covered)/n if len(covered)==n else None))
    return dict(all=summarize(details),by_tier={str(t):summarize([r for r in details if r['tier']==t])
        for t in sorted({r['tier'] for r in details})},
        scope='Full release-record primary and name metrics, including errors. Additional-argument subset explicitly separate. Fixed official primitives with local JSON parsing; not original evaluator/paper equivalence.'),details

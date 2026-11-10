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

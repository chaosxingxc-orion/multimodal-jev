"""One strict answer parser for reward and evaluation.

Invalid syntax is distinct from the legitimate JSON null abstention value.
"""
from __future__ import annotations
import json
import math


class AnswerError(ValueError):
    pass


def _pairs(items):
    result={}
    for key,value in items:
        if key in result:raise AnswerError(f'duplicate JSON key: {key}')
        result[key]=value
    return result


def _constant(value):
    raise AnswerError(f'nonfinite JSON constant: {value}')


def parse_answer(text):
    try:
        answer=json.loads(text,object_pairs_hook=_pairs,parse_constant=_constant)
    except (ValueError,TypeError) as exc:
        raise AnswerError(str(exc)) from exc
    if not isinstance(answer,dict) or 'value' not in answer or set(answer)-{'value','confidence'}:
        raise AnswerError('answer requires value and optional confidence only')
    if 'confidence' in answer:
        c=answer['confidence']
        if type(c) not in (int,float) or not math.isfinite(c) or not 0<=c<=1:
            raise AnswerError('confidence must be finite in [0,1]')
    value=answer['value']
    if type(value) is float and not math.isfinite(value):
        raise AnswerError('nonfinite answer value')
    return answer

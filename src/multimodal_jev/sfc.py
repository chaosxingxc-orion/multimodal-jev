"""SFC call parsing and input separation; no Python calls are ever executed.

The benchmark's original messages (including supplied assistant history) are
retained. Multi-turn examples require multiple ordered audio inputs and cannot
be reduced to the single-audio JevExample contract.
"""
from __future__ import annotations
import ast
from collections import Counter
from copy import deepcopy
import json
import math


def _unique_pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate key')
        result[key] = value
    return result


def _finite(value):
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_finite(v) for v in value)
    if type(value) is dict:
        return all(type(k) is str and _finite(v) for k, v in value.items())
    return False


def parse_calls(text):
    """Strict JSON list of call strings; literal keyword arguments only.

NAN is the literal missing-argument string, never float NaN or JSON null.
Order is preserved here; exact comparison uses a multiset, retaining duplicates.
"""
    try:
        values = json.loads(text, object_pairs_hook=_unique_pairs)
        if type(values) is not list or any(type(v) is not str for v in values):
            raise ValueError('expected a JSON list of function-call strings')
        calls = []
        for value in values:
            node = ast.parse(value, mode='eval').body
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.args:
                raise ValueError('named function and keyword arguments required')
            arguments = {}
            for kw in node.keywords:
                if kw.arg is None or kw.arg in arguments:
                    raise ValueError('duplicate/expanded keyword argument')
                literal = ast.literal_eval(kw.value)
                if not _finite(literal):
                    raise ValueError('non-JSON or nonfinite literal')
                arguments[kw.arg] = literal
            calls.append({'name': node.func.id, 'arguments': arguments})
        return calls
    except (SyntaxError, TypeError) as exc:
        raise ValueError('invalid function-call syntax') from exc


def call_multiset(calls):
    return Counter(json.dumps(call, sort_keys=True, ensure_ascii=False, allow_nan=False) for call in calls)


def strict_metrics(predictions, labels):
    if len(predictions) != len(labels) or not labels:
        raise ValueError('nonempty aligned predictions and labels required')
    counts = {'count': len(labels), 'format_valid': 0, 'exact_correct': 0,
              'function_multiset_correct': 0, 'bound_slot_tp': 0, 'bound_slot_fp': 0, 'bound_slot_fn': 0}
    def slots(calls):
        return Counter((c['name'], key, json.dumps(value, sort_keys=True, ensure_ascii=False))
                       for c in calls for key, value in c['arguments'].items())
    for text, label in zip(predictions, labels):
        gold = parse_calls(label)
        try:
            pred = parse_calls(text)
            valid = True
        except ValueError:
            pred = []
            valid = False
        counts['format_valid'] += valid
        counts['exact_correct'] += valid and call_multiset(pred) == call_multiset(gold)
        counts['function_multiset_correct'] += valid and Counter(c['name'] for c in pred) == Counter(c['name'] for c in gold)
        p, g = slots(pred), slots(gold)
        counts['bound_slot_tp'] += sum((p & g).values())
        counts['bound_slot_fp'] += sum((p - g).values())
        counts['bound_slot_fn'] += sum((g - p).values())
    counts['exact_accuracy'] = counts['exact_correct'] / len(labels)
    counts['format_valid_rate'] = counts['format_valid'] / len(labels)
    tp, fp, fn = (counts[k] for k in ('bound_slot_tp', 'bound_slot_fp', 'bound_slot_fn'))
    counts['bound_slot_f1'] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return counts


def model_input(record):
    """Only the officially supplied conversation and ordered audio paths."""
    return deepcopy({'messages': record['messages'], 'audios': record['audios']})


def source_identity(audio_paths):
    from pathlib import PurePosixPath
    path = PurePosixPath(audio_paths[-1])
    parts = path.parts
    if len(parts) != 5 or parts[:2] != ('audio', 'final_data_audio'):
        raise ValueError('unexpected official audio path')
    level, split = parts[2:4]
    if level not in {'level_1', 'level_2', 'level_3_1', 'level_3_2'} or split not in {'train_id', 'test_id', 'test_ood'}:
        raise ValueError('unknown level/split')
    stem = path.stem
    if level.startswith('level_3'):
        base, stage = stem.rsplit('_', 1)
        if stage not in {'1', '2'}:
            raise ValueError('invalid conversation stage')
    else:
        base, stage = stem, '1'
    group = '/'.join((level, split, base))
    return {'level': level, 'official_split': split, 'source_group': group, 'stage': int(stage)}

"""Strict SFC-Jev outputs, supported-schema validation, and native-score bridge.

Raw output is never repaired. Call assembly does not gate on predicted Noul;
contradictions remain visible as a separate metric.
"""
from collections import Counter
import hashlib
import json

from .consumption import canonical
from .sfc import _finite, call_multiset, parse_calls, strict_metrics
from .sfc_definitions import unique_pairs
from .sfc_jev import restore_bundle


class SchemaError(ValueError):
    """The input schema is unsupported or malformed, not a model error."""


def check_schema(schema):
    if type(schema) is not dict or set(schema) - {'type','description','properties','required','items','enum'}:
        raise SchemaError('unsupported schema keywords')
    kind = schema.get('type')
    if kind not in {'string','integer','number','boolean','null','array','object'}:
        raise SchemaError('unsupported schema type')
    if 'description' in schema and type(schema['description']) is not str:
        raise SchemaError('schema description must be a string')
    if 'enum' in schema and (type(schema['enum']) is not list or not schema['enum'] or not _finite(schema['enum'])):
        raise SchemaError('finite nonempty enum required')
    if kind == 'object':
        props = schema.get('properties'); required = schema.get('required', [])
        if (type(props) is not dict or type(required) is not list
                or any(type(k) is not str for k in required)
                or len(set(required)) != len(required) or not set(required) <= set(props)):
            raise SchemaError('declared object properties and valid required names needed')
        for value in props.values():
            check_schema(value)
    elif 'properties' in schema or 'required' in schema:
        raise SchemaError('object keywords on non-object schema')
    if kind == 'array':
        if 'items' not in schema:
            raise SchemaError('array item schema required')
        check_schema(schema['items'])
    elif 'items' in schema:
        raise SchemaError('array keywords on non-array schema')


def value_errors(value, schema, path='$', allow_missing=False):
    """Closed function arguments; NAN sentinel only at declared parameter positions."""
    if allow_missing and type(value) is str and value == 'NAN':
        return []
    kind = schema['type']
    matches = {'string': type(value) is str, 'integer': type(value) is int,
               'number': type(value) in (int, float), 'boolean': type(value) is bool,
               'null': value is None, 'array': type(value) is list, 'object': type(value) is dict}[kind]
    if not matches or not _finite(value):
        return [path + ': expected ' + kind]
    errors = []
    if 'enum' in schema and canonical(value) not in [canonical(x) for x in schema['enum']]:
        errors.append(path + ': outside enum')
    if kind == 'object':
        props = schema['properties']
        errors.extend(path + '.' + k + ': undeclared' for k in value if k not in props)
        errors.extend(path + '.' + k + ': required' for k in schema.get('required', []) if k not in value)
        for key in value.keys() & props.keys():
            errors.extend(value_errors(value[key], props[key], path + '.' + key, allow_missing=True))
    if kind == 'array':
        for index, item in enumerate(value):
            errors.extend(value_errors(item, schema['items'], f'{path}[{index}]'))
    return errors


def decode(decision, raw_text):
    kind = decision['decision_type']
    if kind == 'typed_call_list':
        check_schema(decision['value_schema'])
    elif kind != 'noul':
        raise SchemaError('unsupported decision type')
    def reject_constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    try:
        if type(raw_text) is not str:
            raise ValueError('raw output must be text')
        value = json.loads(raw_text, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
        if not _finite(value):
            raise ValueError('nonfinite value')
        if kind == 'noul' and type(value) is not bool:
            raise ValueError('expected JSON boolean')
        if kind == 'typed_call_list' and (type(value) is not list or any(type(v) is not dict for v in value)):
            raise ValueError('expected JSON array of argument objects')
    except (ValueError, TypeError, RecursionError) as exc:
        return dict(format_valid=False, schema_valid=False, value=None, errors=[str(exc)])
    errors = value_errors(value, decision['value_schema']) if kind == 'typed_call_list' else []
    return dict(format_valid=True, schema_valid=not errors, value=value, errors=errors)


def _literal(value):
    if type(value) is str:
        return json.dumps(value, ensure_ascii=False)
    if type(value) is list:
        return '[' + ', '.join(_literal(v) for v in value) + ']'
    if type(value) is dict:
        return '{' + ', '.join(_literal(k) + ': ' + _literal(v) for k,v in value.items()) + '}'
    return repr(value)


def native_text(calls):
    if not _finite(calls):
        raise ValueError('finite calls required')
    strings = [c['name'] + '(' + ', '.join(k + '=' + _literal(v) for k,v in c['arguments'].items()) + ')'
               for c in calls]
    result = json.dumps(strings, ensure_ascii=False)
    if call_multiset(parse_calls(result)) != call_multiset(calls):
        raise ValueError('native call serialization must preserve types and multiplicity')
    return result


def score_bundle(request, gold, predictions):
    """All declared decisions required; prediction rows bound to input SHA256."""
    row = restore_bundle(request, gold)
    expected = {d['id'] for d in request['input']['decisions']}
    found = {}
    digest = hashlib.sha256(canonical(request)).hexdigest()
    for prediction in predictions:
        if (set(prediction) != {'example_id','decision_id','request_sha256','raw_text'}
                or prediction['example_id'] != request['id'] or prediction['request_sha256'] != digest
                or prediction['decision_id'] not in expected or prediction['decision_id'] in found
                or type(prediction['raw_text']) is not str):
            raise ValueError('unique input-bound decision predictions required')
        found[prediction['decision_id']] = prediction['raw_text']
    if set(found) != expected:
        raise ValueError('complete decision coverage required')
    counts = Counter(decisions=len(expected), functions=len(expected)//3)
    results = {}; calls = []; all_calls_valid = True
    for decision in request['input']['decisions']:
        key = decision['id']; result = decode(decision, found[key]); results[key] = result
        counts['format_valid'] += result['format_valid']; counts['schema_valid'] += result['schema_valid']
        if decision['decision_type'] == 'noul':
            correct = result['schema_valid'] and result['value'] is gold['answers'][key]
            counts['noul_decisions'] += 1; counts['noul_correct'] += correct
        else:
            pred = [dict(name=decision['function']['name'], arguments=a) for a in result['value']] if result['schema_valid'] else []
            label = [dict(name=decision['function']['name'], arguments=a) for a in gold['answers'][key]]
            correct = result['schema_valid'] and call_multiset(pred) == call_multiset(label)
            counts['call_list_correct'] += correct
            calls.extend(pred); all_calls_valid &= result['schema_valid']
        counts['decision_correct'] += correct
        result['correct'] = bool(correct)
    for decision in request['input']['decisions']:
        if decision['decision_type'] != 'typed_call_list':
            continue
        base = decision['id'].rsplit('.',1)[0]
        a,b,c = (results[base+'.'+suffix] for suffix in ('present','absent','calls'))
        valid_pair = a['schema_valid'] and b['schema_valid']
        counts['valid_noul_pairs'] += valid_pair
        counts['complementary_noul_pairs'] += valid_pair and a['value'] is not b['value']
        valid_triple = valid_pair and c['schema_valid']
        counts['valid_function_triples'] += valid_triple
        counts['consistent_function_triples'] += (valid_triple and a['value'] is bool(c['value']) and b['value'] is (not c['value']))
    converted = native_text(calls) if all_calls_valid else 'INVALID_JEV_CALL_LIST'
    exact = all_calls_valid and call_multiset(calls) == call_multiset(row['supervision']['calls'])
    return dict(counts=dict(counts), decisions=results, native_prediction=converted,
        native_calls_valid=bool(all_calls_valid), native_calls_exact=bool(exact),
        all_decisions_correct=counts['decision_correct']==counts['decisions'],
        strict_native=strict_metrics([converted],[row['supervision']['official_solution']]))

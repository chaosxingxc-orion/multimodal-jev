"""Strict decision scoring with explicit missing supervision and native assembly.

Requires an authenticated task bank, requests, targets and raw-output ledger.
This module scores those values; it does not certify a model execution.
"""
from collections import Counter
import json

from .audio2tool_inference import digest
from .audio2tool_jev import verify_gold
from .audio2tool_score import evaluate_projected, typed


def decode(decision, raw):
    """No fence removal, key repair, bool/number coercion or default arguments."""
    kind = decision['decision_type']
    if kind not in ('noul', 'typed_call_list', 'typed_tool_sequence'):
        raise ValueError('unsupported decision contract')

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result

    def constant(_):
        raise ValueError('nonfinite JSON constant')

    try:
        if type(raw) is not str:
            raise ValueError('raw text required')
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
        typed(value)
        if kind == 'noul' and type(value) is not bool:
            raise ValueError('JSON boolean required')
        if kind == 'typed_call_list' and (
                type(value) is not list or any(type(v) is not dict for v in value)):
            raise ValueError('array of argument objects required')
        if kind == 'typed_tool_sequence' and (
                type(value) is not list or any(type(v) is not str or
                    v not in decision['value_schema']['items']['enum'] for v in value)):
            raise ValueError('array of registered tool names required')
    except (ValueError, RecursionError) as exc:
        return dict(valid=False, value=None, error=str(exc))
    return dict(valid=True, value=value, error=None)


def assemble(bank, results):
    """Use predicted sequence and lists only; Noul never filters/repairs calls."""
    sequence = results['tools.sequence']
    if not sequence['valid']:
        return None
    lists = {}
    for tool in bank['tools']:
        result = results['tool.' + tool['tool_id'] + '.calls']
        if not result['valid']:
            return None
        lists[tool['tool_name']] = result['value']
    counts = Counter(sequence['value'])
    if any(counts[name] != len(values) for name, values in lists.items()):
        return None
    positions = Counter()
    calls = []
    for name in sequence['value']:
        calls.append(dict(name=name, arguments=lists[name][positions[name]]))
        positions[name] += 1
    return calls


def score_bundle(request, gold, bank, predictions, official):
    labels = verify_gold(request, gold, bank, official)
    expected = {d['id'] for d in bank['decisions']}
    found = {}
    request_sha = digest(request)
    for p in predictions:
        if (set(p) != {'example_id', 'decision_id', 'request_sha256', 'raw_text', 'error'}
                or p['example_id'] != request['id'] or p['request_sha256'] != request_sha
                or p['decision_id'] not in expected or p['decision_id'] in found
                or type(p['raw_text']) is not str
                or (p['error'] is not None and type(p['error']) is not str)):
            raise ValueError('unique input-bound raw terminal decisions required')
        found[p['decision_id']] = p
    if set(found) != expected:
        raise ValueError('complete decision coverage required, including unavailable gold')
    counts = Counter(decisions=len(expected), tools=len(bank['tools']))
    results = {}
    for decision in bank['decisions']:
        key = decision['id']; p = found[key]; label = labels[key]
        kind = decision['decision_type']
        result = (decode(decision, p['raw_text']) if p['error'] is None else
                  dict(valid=False, value=None, error=p['error']))
        available = label['status'] == 'available'
        correct = bool(result['valid'] and typed(result['value']) == typed(label['value'])) if available else None
        result.update(supervision_available=available, correct=correct)
        results[key] = result
        counts['runtime_errors'] += p['error'] is not None
        counts['format_valid'] += result['valid']
        counts[kind + '_decisions'] += 1
        counts[kind + '_available'] += available
        counts[kind + '_unavailable'] += not available
        counts[kind + '_correct'] += correct is True
    # Keep semantic presence separate for each polarity. Invalid decisions never
    # become correct negative answers; both true classes retain full denominators.
    for tool in bank['tools']:
        base = 'tool.' + tool['tool_id']
        a, b, c = (results[base + '.' + suffix] for suffix in ('present', 'absent', 'calls'))
        truth = labels[base + '.present']['value']
        for head, result in [('present', a), ('absent', b)]:
            prefix = head + '_semantic_'
            if not result['valid']:
                cell = 'invalid_positive' if truth else 'invalid_negative'
            else:
                positive = result['value'] if head == 'present' else not result['value']
                cell = ('tp' if truth else 'fp') if positive else ('fn' if truth else 'tn')
            counts[prefix + cell] += 1
        valid_pair = a['valid'] and b['valid']
        complementary = valid_pair and a['value'] is not b['value']
        counts['valid_noul_pairs'] += valid_pair
        counts['complementary_noul_pairs'] += complementary
        counts['consistent_tool_triples'] += (complementary and c['valid'] and a['value'] is bool(c['value']))
    calls = assemble(bank, results)
    consistent = calls is not None and counts['consistent_tool_triples'] == len(bank['tools'])
    native_prediction = dict(id=request['id'], raw_text=json.dumps(calls, ensure_ascii=False) if calls is not None else '',
                             error=None if calls is not None else 'invalid_or_inconsistent_jev_call_assembly')
    return dict(id=request['id'], tier=request['tier'], counts=dict(counts), decisions=results,
                native_prediction=native_prediction, native_assembly_valid=calls is not None,
                all_predictions_consistent=consistent)


def summarize(counts):
    c = Counter(counts)
    ratio = lambda a, b: a / b if b else None
    result = dict(counts=dict(c), format_valid_rate=ratio(c['format_valid'], c['decisions']),
                  decision_accuracy={kind:ratio(c[kind + '_correct'], c[kind + '_available'])
                    for kind in ('noul', 'typed_call_list', 'typed_tool_sequence')},
                  complementary_pair_rate=ratio(c['complementary_noul_pairs'], c['tools']),
                  consistent_tool_triple_rate=ratio(c['consistent_tool_triples'], c['tools']))
    result['semantic_presence'] = {}
    for head in ('present', 'absent'):
        cells = {k:c[head + '_semantic_' + k] for k in ('tp', 'fp', 'fn', 'tn', 'invalid_positive', 'invalid_negative')}
        tp, fp, fn, tn, ip, ine = cells.values()
        recall = ratio(tp, tp + fn + ip); specificity = ratio(tn, tn + fp + ine)
        result['semantic_presence'][head] = dict(cells,
            precision=ratio(tp, tp + fp), recall=recall, specificity=specificity,
            f1=ratio(2 * tp, 2 * tp + fp + fn + ip),
            balanced_accuracy=(recall + specificity) / 2 if recall is not None and specificity is not None else None,
            valid_rate=ratio(tp + fp + fn + tn, sum(cells.values())))
    return result


def evaluate(bundles, bank, official):
    """Stream (request, gold, 457 terminal predictions) tuples; keep compact totals.

    Caller must verify this stream covers its predeclared full population. This
    evaluator rejects duplicate sample IDs but cannot infer omitted input rows.
    """
    seen = set(); total = Counter(); tiers = {}; targets = []; native = []
    for request, gold, predictions in bundles:
        if request['id'] in seen:
            raise ValueError('duplicate release record')
        seen.add(request['id'])
        scored = score_bundle(request, gold, bank, predictions, official)
        c = Counter(scored['counts']); c['records'] = 1
        c['native_assembly_valid'] = scored['native_assembly_valid']
        c['all_predictions_consistent'] = scored['all_predictions_consistent']
        total.update(c); tiers.setdefault(str(request['tier']), Counter()).update(c)
        targets.append(gold['native_target']); native.append(scored['native_prediction'])
    if not seen:
        raise ValueError('nonempty declared population required')
    native_report, _ = evaluate_projected(targets, native, official)
    return dict(all=summarize(total), by_tier={k:summarize(v) for k,v in sorted(tiers.items())},
        native=native_report,
        scope='Strict raw Jev decisions on supplied population. Missing gold excluded only from its exact-match denominator; prediction coverage remains required. Native assembly uses lists and sequence independently of Noul. This scorer does not authenticate inference provenance or claim unchanged paper protocol.')

"""Independent raw JSON/count audit for the frozen FSC six-view report.

Does not call production answer parsing, scoring or paired-statistics helpers.
Caller authenticates the frozen task release and complete successful runs.
"""
from collections import defaultdict
import hashlib
import json
import math

VIEWS = ('native', 'permutation', 'rebind', 'missing_candidate', 'noul.True', 'noul.False')
FIELDS = ('both_correct', 'both_wrong', 'fixed', 'regressed')


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate answer key')
            result[key] = value
        return result

    def constant(value):
        raise ValueError('nonfinite answer')

    answer = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    if type(answer) is not dict or 'value' not in answer or set(answer)-{'value', 'confidence'}:
        raise ValueError('strict answer object required')
    if 'confidence' in answer:
        confidence = answer['confidence']
        if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
            raise ValueError('invalid confidence')
    value = answer['value']
    if type(value) is float and not math.isfinite(value):
        raise ValueError('nonfinite value')
    return value


def score(requests, targets, predictions):
    gold = {r['id']: r for r in targets}
    indexed = {r['example_id']: r for r in predictions}
    ids = {r['id'] for r in requests}
    if (not requests or len(ids) != len(requests) or len(gold) != len(targets)
            or len(indexed) != len(predictions) or ids != set(gold) or ids != set(indexed)):
        raise ValueError('exact complete unique request/target/prediction roster required')
    groups = defaultdict(dict); audio = {}; splits = set()
    cells = {v: dict(examples=0, correct=0, format_valid=0, schema_valid=0) for v in VIEWS}
    for request in requests:
        uid = request['id']; target = gold[uid]; pred = indexed[uid]
        group = request['source_group']; view = target['metadata']['variant']
        pin = hashlib.sha256((json.dumps(request,sort_keys=True,ensure_ascii=False)+'\n').encode()).hexdigest()
        if (target['request_sha256'] != pin or pred['request_sha256'] != pin
                or type(pred['replicate']) is not int or pred['replicate'] != 0
                or view not in VIEWS or view in groups[group]
                or target['metadata']['source_group'] != group
                or target['metadata']['audio_sha256'] != request['audio_sha256']):
            raise ValueError('original binding, single replicate and unique views required')
        if group in audio and audio[group] != request['audio_sha256']:
            raise ValueError('one recording per group required')
        audio[group] = request['audio_sha256']; splits.add(request['split'])
        schema = request['input']['schema']
        if schema['decision_type'] not in ('choice', 'noul'):
            raise ValueError('only frozen FSC choice/noul tasks supported')
        try:
            value = parse(pred['raw_text']); valid = True
        except (ValueError, TypeError):
            value = None; valid = False
        accepted = valid and ((value is None and schema['allow_unknown']) or
            (schema['decision_type'] == 'choice' and type(value) is str
             and value in {c['id'] for c in schema['candidates']}) or
            (schema['decision_type'] == 'noul' and type(value) is bool))
        correct = valid and type(value) is type(target['target']) and value == target['target']
        for field, computed in [('format_valid', valid), ('schema_valid', accepted)]:
            if field in pred and (type(pred[field]) is not bool or pred[field] != computed):
                raise ValueError('stored prediction flags differ from raw response')
        cell = cells[view]; cell['examples'] += 1; cell['correct'] += correct
        cell['format_valid'] += valid; cell['schema_valid'] += accepted
        groups[group][view] = correct
    if len(splits) != 1 or any(set(g) != set(VIEWS) for g in groups.values()):
        raise ValueError('single split with six complete views per recording required')
    count = len(groups)
    for cell in cells.values():
        cell['accuracy'] = cell['correct']/count
    pair_counts = {}
    for view in ('permutation', 'rebind', 'missing_candidate', 'noul'):
        correct = sum(g['noul.True'] and g['noul.False'] if view == 'noul'
                      else g['native'] and g[view] for g in groups.values())
        pair_counts[view] = dict(groups=count,both_correct=correct,both_correct_rate=correct/count)
    all_six = sum(all(g.values()) for g in groups.values())
    metrics = dict(replicate=0,by_variant=cells,pairs=pair_counts,all_six_correct=all_six,
                   all_six_correct_rate=all_six/count)
    return dict(source_groups=count,requests=len(requests),metrics=metrics,outcomes=dict(groups))


def exact_p(fixed, regressed):
    """Independent integer binomial arithmetic; final float can underflow."""
    if any(type(n) is not int or n < 0 for n in (fixed, regressed)):
        raise ValueError('nonnegative integer discordant counts required')
    n = fixed+regressed
    numerator = 2*sum(math.comb(n,i) for i in range(min(fixed,regressed)+1))
    denominator = 1 << n
    return min(numerator,denominator)/denominator


def transitions(parent, candidate):
    if not parent or set(parent) != set(candidate):
        raise ValueError('same complete recording roster required')
    by_view = {v: dict.fromkeys(FIELDS,0) for v in VIEWS}
    whole = dict.fromkeys(FIELDS,0)

    def count(cell, a, b):
        cell['both_correct' if a and b else 'both_wrong' if not a and not b
             else 'fixed' if b else 'regressed'] += 1

    for group, old in parent.items():
        new = candidate[group]
        if (set(old) != set(VIEWS) or set(new) != set(VIEWS)
                or any(type(v) is not bool for v in [*old.values(),*new.values()])):
            raise ValueError('six boolean outcomes per recording required')
        for view in VIEWS:
            count(by_view[view],old[view],new[view])
        count(whole,all(old.values()),all(new.values()))
    for cell in [*by_view.values(),whole]:
        cell['mcnemar_exact_two_sided_p'] = exact_p(cell['fixed'],cell['regressed'])
    return dict(by_view=by_view,all_six_by_source_group=whole)


def require_same_transitions(actual, expected):
    for key in ('by_view', 'all_six_by_source_group'):
        if key == 'by_view':
            if set(actual[key]) != set(expected[key]):
                raise ValueError('paired view roster differs')
            pairs = [(actual[key][v],expected[key][v]) for v in VIEWS]
        else:
            pairs = [(actual[key],expected[key])]
        for a,b in pairs:
            if any(a[f] != b[f] for f in FIELDS) or not math.isclose(
                    a['mcnemar_exact_two_sided_p'],b['mcnemar_exact_two_sided_p'],
                    rel_tol=1e-9,abs_tol=1e-12):
                raise ValueError('paired counts or exact p-value differs')

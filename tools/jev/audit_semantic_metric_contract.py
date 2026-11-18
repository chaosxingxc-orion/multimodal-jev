#!/usr/bin/env python3
"""Executable metric definitions on synthetic cases only; no model scorer entry."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

STATES = ('positive', 'negative', 'unavailable', 'conflict', 'unassessed')


def divide(a, b):
    return a / b if b else None


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.


def binary_summary(rows):
    """Rows are (explicit supervision state, typed prediction), never native labels."""
    counts = Counter({k: 0 for k in ('tp', 'fp', 'fn', 'tn', 'invalid_positive',
                                     'invalid_negative', 'invalid_all', *STATES)})
    for state, prediction in rows:
        if state not in STATES:
            raise ValueError('explicit supervision state required')
        counts[state] += 1
        valid = type(prediction) is bool
        counts['invalid_all'] += not valid
        if state not in ('positive', 'negative'):
            continue
        positive = state == 'positive'
        if not valid:
            counts['invalid_positive' if positive else 'invalid_negative'] += 1
            counts['fn'] += positive
        else:
            counts['tp' if positive and prediction else 'fn' if positive else
                   'fp' if prediction else 'tn'] += 1
    p, n = counts['positive'], counts['negative']
    recall = divide(counts['tp'], p)
    specificity = divide(counts['tn'], n)
    return dict(counts=dict(counts), coordinates=len(rows), admitted=p+n,
        coverage=divide(p+n, len(rows)), positive_recall=recall,
        false_positive_rate=divide(counts['fp'], n),
        negative_failure_rate=divide(counts['fp']+counts['invalid_negative'], n),
        specificity=specificity,
        balanced_accuracy=(recall+specificity)/2 if p and n else None,
        positive_f1=f1(counts['tp'], counts['fp'], counts['fn']) if p+n else None,
        invalid_rate_all=divide(counts['invalid_all'], len(rows)),
        invalid_rate_admitted=divide(counts['invalid_positive']+counts['invalid_negative'], p+n))


def native_summary(classes, rows):
    """Rows: (gold canonical class, prediction or invalid value, source group)."""
    if not classes or len(classes) != len(set(classes)) or not rows:
        raise ValueError('nonempty fixed classes and population required')
    support = Counter(); predicted = Counter(); correct = Counter(); groups = defaultdict(list)
    invalid = 0
    for gold, prediction, group in rows:
        if gold not in classes or not isinstance(group, str) or not group:
            raise ValueError('valid gold and source identity required')
        valid = type(prediction) is str and prediction in classes
        support[gold] += 1
        invalid += not valid
        if valid:
            predicted[prediction] += 1
        hit = valid and prediction == gold
        correct[gold] += hit; groups[group].append(hit)
    per_class = {c: dict(support=support[c], predicted=predicted[c], correct=correct[c],
        f1=f1(correct[c], predicted[c]-correct[c], support[c]-correct[c]),
        recall=divide(correct[c], support[c])) for c in classes}
    present = [c for c in classes if support[c]]
    return dict(recordings=len(rows), accuracy=sum(correct.values())/len(rows),
        invalid_rate=invalid/len(rows), per_class=per_class,
        macro_f1_fixed=sum(v['f1'] for v in per_class.values())/len(classes),
        macro_f1_supported=sum(per_class[c]['f1'] for c in present)/len(present),
        macro_recall_supported=sum(per_class[c]['recall'] for c in present)/len(present),
        supported_classes=len(present),
        source_equal_accuracy=sum(sum(v)/len(v) for v in groups.values())/len(groups))


def panel_summary(definitions, rows):
    """One slot panel; rows have complete state maps and possibly missing predictions."""
    if len(set(definitions)) != len(definitions):
        raise ValueError('unique configured definitions required')
    for states, predictions in rows:
        if set(states) != set(definitions) or not set(predictions) <= set(definitions):
            raise ValueError('exact state map and known prediction keys required')
        if any(s not in STATES for s in states.values()):
            raise ValueError('explicit supervision state required')
    by_definition = {d: binary_summary([(s[d], p.get(d)) for s, p in rows]) for d in definitions}
    # Zero-slot panels are inapplicable, never vacuously perfect empty predictions.
    complete = [(s, p) for s, p in rows if definitions and
                all(v in ('positive', 'negative') for v in s.values())]
    complete_scores = {d: binary_summary([(s[d], p.get(d)) for s, p in complete]) for d in definitions}
    exact = sum(all(type(p.get(d)) is bool and p[d] == (s[d] == 'positive')
                    for d in definitions) for s, p in complete)
    invalid = sum(any(type(p.get(d)) is not bool for d in definitions) for s, p in complete)
    c = Counter()
    for result in complete_scores.values():
        c.update(result['counts'])
    supported = [v['positive_f1'] for v in complete_scores.values() if v['counts']['positive']]
    return dict(configured_definitions=len(definitions), recordings=len(rows),
        decision_metrics=by_definition, complete_recordings=len(complete),
        complete_coverage=divide(len(complete), len(rows)) if definitions else None,
        complete_set_em=divide(exact, len(complete)), complete_invalid_rate=divide(invalid, len(complete)),
        complete_micro_positive_f1=f1(c['tp'], c['fp'], c['fn']) if complete else None,
        complete_macro_positive_f1=divide(sum(supported), len(supported)),
        complete_supported_definitions=len(supported),
        inapplicable_no_slot_schema=not definitions)


def audit():
    native = native_summary(['a', 'b', 'c'], [('a', 'a', 'g1'), ('b', 'a', 'g1'), ('a', None, 'g2')])
    invalid = binary_summary([('positive', None), ('negative', None)])
    partial = panel_summary(['s1', 's2'], [({'s1': 'positive', 's2': 'unassessed'}, {'s1': True, 's2': True})])
    empty = panel_summary(['s1'], [({'s1': 'negative'}, {})])
    no_slot = panel_summary([], [({}, {})])
    return dict(schema='semantic-metric-contract-audit-v2', cases=dict(
        native_fixed_vs_supported_macro=native, invalid_binary_outputs=invalid,
        partially_supervised_panel=partial, missing_empty_panel_prediction=empty,
        no_slot_panel=no_slot),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        training_dispatch_allowed=False, evaluation_dispatch_allowed=False,
        scope='Synthetic metric arithmetic only; no dataset labels, predictions, scorer integration, statistical CI, or admission.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(); args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'cases': list(report['cases']), 'output': str(args.output)}))

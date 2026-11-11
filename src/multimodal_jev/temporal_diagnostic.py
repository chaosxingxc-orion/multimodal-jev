"""Strict boolean and complete-bundle scoring for the temporal pilot."""
from collections import defaultdict

from .answers import AnswerError, parse_answer


def score(targets, predictions):
    gold = {r['example_id']: r for r in targets}
    if len(gold) != len(targets) or not gold:
        raise ValueError('nonempty unique targets required')
    pred = {r['example_id']: r for r in predictions}
    if len(pred) != len(predictions) or set(pred) != set(gold):
        raise ValueError('exact unique prediction coverage required')
    by_task, bundles = defaultdict(list), defaultdict(dict)
    for uid, row in gold.items():
        if type(row['target']) is not bool:
            raise ValueError('boolean gold required')
        valid, value = False, None
        try:
            value = parse_answer(pred[uid]['raw_text'])['value']
            valid = type(value) is bool
        except AnswerError:
            pass
        correct = valid and value == row['target']
        source, task = uid.rsplit('::', 1)
        if task not in {'date', 'time', 'timeofday'}:
            raise ValueError('unexpected temporal task')
        by_task[task].append(dict(valid=valid, correct=correct, gold=row['target'], value=value))
        bundles[source][task] = correct
    def summary(rows):
        return dict(total=len(rows), correct=sum(r['correct'] for r in rows),
                    valid_boolean=sum(r['valid'] for r in rows),
                    predicted_true=sum(r['valid'] and r['value'] is True for r in rows),
                    positive_total=sum(r['gold'] for r in rows),
                    positive_correct=sum(r['gold'] and r['correct'] for r in rows),
                    negative_correct=sum(not r['gold'] and r['correct'] for r in rows))
    complete = [r for r in bundles.values() if set(r) == {'date','time','timeofday'}]
    return dict(decisions=summary([r for rows in by_task.values() for r in rows]),
                by_task={k:summary(v) for k,v in sorted(by_task.items())},
                complete_bundles=dict(total=len(complete),correct=sum(all(r.values()) for r in complete)),
                partial_bundles=len(bundles)-len(complete))

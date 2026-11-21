import json
from pathlib import Path

import pytest

from tools.reports.build_jev_evidence_panel import scan_calendar
from tools.reports.score_jev_boundary_panel import calendar_contrasts


def test_calendar_grammar_keeps_only_supported_unambiguous_questions():
    path = Path(__file__).resolve().parents[1] / 'configs/portable/operations/jev-evidence-calendar-train-v1.json'
    rules = json.loads(path.read_text())['calendar']
    examples = [
        ('a', 'calendar_query', 'do i have a meeting today'),
        ('b', 'calendar_query', 'are there any meetings scheduled tomorrow'),
        ('c', 'calendar_query', 'who am i meeting tomorrow'),
        ('d', 'calendar_query', 'do i have a meeting today or tomorrow'),
        ('e', 'calendar_query', 'am i meeting anybody today'),
        ('f', 'calendar_set', 'am i meeting anybody today'),
    ]
    rows = [dict(id=uid, gold=label, text=text) for uid, label, text in reversed(examples)]
    selected, counts = scan_calendar(rows, rules)
    assert [(row['id'], day) for row, day in selected] == [('a', 0), ('b', 1)]
    assert counts == dict(train_recordings=6, native_intent_recordings=5,
                          outside_grammar=2, ambiguous=1, today=1, tomorrow=1)


def test_calendar_missing_answer_is_masked_and_failures_require_correct_baseline():
    def gold(availability):
        return dict(intent=True, intent_sufficient=True, availability=availability,
                    availability_sufficient=availability is not None)

    def prob(answer, sufficient):
        return dict(intent=.9, intent_sufficient=.9, availability=answer, availability_sufficient=sufficient)

    ids = ('complete', 'unsupported', 'baseline')
    targets = {arm: [dict(id=uid, gold=gold(answer)) for uid in ids]
               for arm, answer in dict(forward=True, reverse=False, irrelevant=True, missing=None).items()}
    predictions = dict(
        forward=dict(zip(ids, [prob(.9, .9), prob(.9, .9), prob(.1, .9)])),
        reverse={uid: prob(.1, .9) for uid in ids},
        irrelevant={uid: prob(.9, .9) for uid in ids},
        missing={'unsupported': prob(.9, .9), 'baseline': prob(.9, .9), 'complete': prob(.9, .1)},
    )
    report = calendar_contrasts(targets, predictions, .5)
    assert report['recording_ids']['missing_wrong'] == ['unsupported']
    assert report['recording_ids']['baseline_wrong'] == ['baseline']
    assert report['recording_ids']['all_correct'] == ['complete']
    assert report['per_arm']['missing']['unknown_recall'] == pytest.approx(1 / 3)
    del predictions['reverse']['complete']
    with pytest.raises(ValueError, match='complete calendar predictions'):
        calendar_contrasts(targets, predictions, .5)

import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/jev'))
from build_semantic_selection_design import requests, roster


def row(identity, role):
    return dict(id=identity, audio=identity+'.flac', audio_sha256=identity+'hash',
                source_group=identity+'group', role=role, official_split='train',
                known_pending_source=False, intent='secret', transcript='secret')


def panel():
    questions = [dict(operator='I_present', definition_id='x')]
    return dict(panel_id='semantic-16', process_questions=questions,
                arms=dict(Direct=[], Q_input=questions, Q_credit=questions))


def test_allowlisted_full_population_not_gold_routing():
    selection = [row('b', 'selection'), row('a', 'selection')]
    learning = [row('c', 'learning')]
    original = roster(selection, learning)
    changed = copy.deepcopy(selection)
    changed[0].update(intent='other', transcript='other', semantic_label=True)
    assert original == roster(changed, learning)
    assert [r['id'] for r in original] == ['a', 'b']
    assert all(set(r) == {'id', 'audio', 'audio_sha256', 'source_group'} for r in original)
    schedule = list(requests(original, [panel()]))
    assert len(schedule) == len({r['request_id'] for r in schedule}) == 6
    assert {r['arm'] for r in schedule} == {'Direct', 'Q_input', 'Q_credit'}
    assert all(r['transform'] == 'canonical' for r in schedule)


@pytest.mark.parametrize('field', ['id', 'source_group', 'audio_sha256'])
def test_split_leakage_rejected(field):
    s = row('a', 'selection'); t = row('b', 'learning'); t[field] = s[field]
    with pytest.raises(ValueError, match='overlap'):
        roster([s], [t])


def test_reject_historical_or_nonpaired_questions_and_duplicate_population():
    p = panel(); p['process_questions'][0]['operator'] = 'I_class'
    with pytest.raises(ValueError, match='historical'):
        list(requests([row('a', 'selection')], [p]))
    p = panel(); p['arms']['Q_credit'] = []
    with pytest.raises(ValueError, match='paired'):
        list(requests([row('a', 'selection')], [p]))
    with pytest.raises(ValueError, match='unique'):
        roster([row('a', 'selection')]*2, [row('b', 'learning')])
    s = row('a', 'selection'); s['known_pending_source'] = True
    with pytest.raises(ValueError, match='role'):
        roster([s], [row('b', 'learning')])

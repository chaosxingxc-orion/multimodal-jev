import math

import pytest

from multimodal_jev.card_calibrated_mining import mine


def test_calibrated_mining_prioritizes_rank_and_preserves_replay_ratio():
    probabilities = {
        'wrong': [.6, .8],
        'tie': [.8, .8],
        'high_negative': [.99, .98],
        'boundary_also_low': [.80, .78],
        'low': [.7, .2],
        'low_spare': [.75, .2],
        'clear': [.98, .2],
    }
    clean = {uid: dict(intent='gold', group=uid, text=uid, synthetic=False)
             for uid in probabilities}
    policy = dict(a=2, b=-1, confidence=.9, boundary_gap=.1,
                  replay_fraction=.2, seed=317)
    rows, audit = mine(clean, probabilities, ['gold', 'rival'], policy)

    assert rows['wrong']['stratum'] == 'error'
    assert rows['tie']['stratum'] == 'tie'
    assert rows['high_negative']['stratum'] == 'high_negative'
    assert rows['boundary_also_low']['stratum'] == 'boundary'
    assert rows['boundary_also_low']['ref_gap'] == pytest.approx(.02)
    assert rows['boundary_also_low']['calibrated_gap'] < .1
    assert rows['clear']['stratum'] == 'replay'
    assert sum(row['stratum'] == 'low_positive' for row in rows.values()) == 0
    assert (audit['hard'], audit['replay_actual']) == (4, 1)
    assert rows['wrong']['calibrated_gold'] == pytest.approx(1 / (1 + math.exp(1) * (2/3)**2))
    assert rows == mine(dict(reversed(list(clean.items()))), probabilities, ['gold', 'rival'], policy)[0]

    # Three primary rows require one low-confidence row to reach a 4:1 ratio.
    subset = {uid: row for uid, row in clean.items() if uid != 'tie'}
    _, fewer = mine(subset, probabilities, ['gold', 'rival'], policy)
    assert (fewer['primary_hard'], fewer['low_positive_rounding'], fewer['replay_actual']) == (3, 1, 1)
    with pytest.raises(ValueError, match='insufficient'):
        mine({k:v for k,v in subset.items() if k != 'clear'}, probabilities, ['gold', 'rival'], policy)

from fractions import Fraction as F
from pathlib import Path
import runpy

import pytest

api = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/jev/audit_semantic_reward_redundancy.py'))


def test_selective_duplication_changes_preference_despite_fixed_total_cap():
    r = api['audit']()
    assert list(map(F, r['before'])) == [F(1, 30), F(1, 15)]
    assert list(map(F, r['after'])) == [F(3, 50), F(1, 25)]
    assert max(map(F, r['after'])) < F(r['process_cap'])


def test_global_replication_is_different_from_selective_replication():
    reward = api['mean_credit']
    original = reward([True, False], [True, True])
    assert reward([True, False] * 3, [True] * 6) == original
    assert reward([True, True, False], [True] * 3) > original


def test_masked_duplicate_does_not_reallocate_budget():
    reward = api['mean_credit']
    assert reward([True, True, False], [True, False, True]) == F(1, 30)
    assert reward([True, False, False], [True, False, True]) == F(1, 30)


def test_invalid_masks_rejected():
    with pytest.raises(ValueError):
        api['mean_credit']([True], [])
    with pytest.raises(ValueError):
        api['mean_credit']([True], [None])

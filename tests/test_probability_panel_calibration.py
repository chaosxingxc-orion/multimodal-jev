import numpy as np
import pytest

from tools.reports.calibrate_tpo_probabilities import routing, thresholds


def test_threshold_excludes_unsupported_class_and_counts_rejection_as_miss():
    raw = [[.91, .06, .02], [.54, .36, .07], [.05, .86, .04], [.03, .20, .73]]
    gold = [0, 1, 1, 1]
    fitted = thresholds(raw, gold)
    assert fitted == dict(threshold=.86, macro_f1=.75, queries=4, supported_classes=2)
    scores = np.array(raw)
    counts = routing(scores.max(axis=1), scores.argmax(axis=1) == gold, fitted["threshold"])
    assert counts["coverage"] == .5
    assert counts["accepted_error_rate"] == 0
    assert counts["correct_top1_rejection_rate"] == 0


def test_threshold_groups_exact_score_ties_and_uses_the_highest_tied_cut():
    raw = [[.8, .1], [.2, .8], [.1, .6]]
    assert thresholds(raw, [0, 1, 0])["threshold"] == .8
    # No correct Top-1 label: all observed thresholds tie at zero.
    assert thresholds(raw, [1, 0, 0])["threshold"] == 1
    with pytest.raises(ValueError, match="probability"):
        thresholds([[float("nan"), .2]], [1])

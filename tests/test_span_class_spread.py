from collections import Counter

import pytest

from multimodal_jev.span_class_spread import class_spread_indices


def test_unequal_classes_remain_spread_across_epoch():
    labels = ['head'] * 1000 + ['middle'] * 100 + ['tail'] * 10
    indices = class_spread_indices(labels, seed=317, epoch=0)
    actual = [labels[i] for i in indices]
    assert Counter(actual) == Counter(labels)
    assert sorted(indices) == list(range(len(labels)))
    for start in range(0, 1110, 111):
        counts = Counter(actual[start:start + 111])
        assert abs(counts['head'] - 100) <= 2
        assert abs(counts['middle'] - 10) <= 2
        assert counts['tail'] >= 1
    # The old exhausted round-robin tail would contain only head examples.
    assert set(actual[-111:]) == set(labels)


def test_seed_replay_and_payload_alignment():
    labels = ['a'] * 13 + ['b'] * 7 + ['c'] * 4
    indices = class_spread_indices(labels, seed=317, epoch=0)
    assert indices == class_spread_indices(labels, seed=317, epoch=0)
    assert indices != class_spread_indices(labels, seed=317, epoch=1)
    payload = [dict(id=i, weight=1/(1+i), label=label) for i,label in enumerate(labels)]
    ordered = [payload[i] for i in indices]
    assert sorted(ordered, key=lambda row: row['id']) == payload
    for label in set(labels):
        assert [i for i in indices if labels[i] == label] == [i for i,l in enumerate(labels) if l == label]


@pytest.mark.parametrize('labels,seed,epoch', [([], 1, 0), ([''], 1, 0), ([None], 1, 0), (['a'], True, 0), (['a'], 1, -1)])
def test_reject_invalid(labels, seed, epoch):
    with pytest.raises(ValueError):
        class_spread_indices(labels, seed=seed, epoch=epoch)

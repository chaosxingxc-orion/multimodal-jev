import pytest

from multimodal_jev.span_length_groups import length_group_indices


@pytest.mark.parametrize('world', [1, 4, 8])
@pytest.mark.parametrize('micro', [1, 8, 16])
def test_fixed_population_and_capacity(world, micro):
    lengths = [(2000 + i % 17, 100 + (i * 31) % 800) for i in range(128)]
    ranks = length_group_indices(lengths, microbatch=micro, world_size=world, order='length_grouped')
    assert all(len(r) == 128 // world for r in ranks)
    assert sorted(i for r in ranks for i in r) == list(range(128))
    assert ranks == length_group_indices(lengths, microbatch=micro, world_size=world, order='length_grouped')
    for rank in ranks:
        for j in range(0, len(rank), micro):
            group = rank[j:j+micro]
            assert group == sorted(group, key=lambda i: (*lengths[i], i))


def test_permutation_keeps_entire_payload_and_reduces_padding():
    # Each object stands for an indivisible request+G16+labels+weights payload.
    payload = [object() for _ in range(128)]
    lengths = [(1960 + 100 * (i % 2), 100 + 900 * (i % 2)) for i in range(128)]
    original = length_group_indices(lengths, microbatch=8, world_size=4, order='original')
    grouped = length_group_indices(lengths, microbatch=8, world_size=4, order='length_grouped')
    def slots(ranks):
        return sum(8 * max(lengths[i][1] for i in rank[j:j+8])
                   for rank in ranks for j in range(0, len(rank), 8))
    assert slots(grouped) < slots(original)
    restored = [None] * 128
    for rank in grouped:
        for i in rank:
            restored[i] = payload[i]
    assert all(a is b for a, b in zip(restored, payload))


@pytest.mark.parametrize('lengths,micro,world', [([], 1, 1), ([(2, 3)] * 3, 2, 1),
                                               ([(True, 3)], 1, 1), ([(2, 0)], 1, 1)])
def test_invalid_plan(lengths, micro, world):
    with pytest.raises(ValueError):
        length_group_indices(lengths, microbatch=micro, world_size=world, order='length_grouped')

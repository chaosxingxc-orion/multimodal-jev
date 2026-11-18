"""Task-schedule invariants, independent of model execution or gold labels."""
import runpy
from pathlib import Path

import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/audit_slurp_menu_design.py'))
panel = MODULE['panel']
ORDERS = {'intent': [f'i{x}' for x in range(60)], 'slot': [f's{x}' for x in range(55)]}


def test_every_source_can_eventually_receive_every_atomic_definition():
    for group in ('source-a', 'source-b', 'source-c'):
        for family, order in ORDERS.items():
            seen = [panel(ORDERS, group, epoch, 317, 7)[family][0] for epoch in range(len(order))]
            assert len(set(seen)) == len(order)
            assert set(seen) == set(order)


def test_old_stride_four_leaves_three_quarters_of_intents_unseen():
    seen = {panel(ORDERS, 'source-a', epoch, 317, 4)['intent'][0] for epoch in range(60)}
    assert len(seen) == 15


def test_recording_identity_and_label_cannot_route_menu():
    recordings = [{'id': 'mic1', 'source_group': 'shared', 'gold': 'a'},
                  {'id': 'mic2', 'source_group': 'shared', 'gold': 'b'}]
    selected = [panel(ORDERS, r['source_group'], 0, 317, 7) for r in recordings]
    assert selected[0] == selected[1]
    with pytest.raises(TypeError):
        panel(ORDERS, 'shared', 0, 317, 7, gold='a')


def test_two_epoch_panels_are_nonoverlapping_and_restart_deterministic():
    for family in ORDERS:
        first = panel(ORDERS, 'source-a', 0, 317, 7)[family]
        second = panel(ORDERS, 'source-a', 1, 317, 7)[family]
        assert len(first) == len(set(first)) == 4
        assert not set(first) & set(second)
        assert first == panel(ORDERS, 'source-a', 0, 317, 7)[family]

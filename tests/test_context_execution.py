import hashlib
import json
import time

import pytest

from multimodal_jev.context_execution import (
    Stats, content_key, map_cpu, map_io, singleflight, width, widths)


def double(value):
    return value * 2


def identity_row(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def fail_on(value):
    if value == 3:
        raise ValueError('injected failure')
    return value


def slow(value):
    time.sleep(0.01)
    return value


def test_widths_read_from_config_and_reject_unknown_keys():
    assert widths(dict(io_workers=4, cpu_workers=2)) == dict(io_workers=4, cpu_workers=2)
    for bad in [dict(io_workers=4), dict(io_workers=4, cpu_workers=2, extra=1),
                dict(io_workers=0, cpu_workers=2), dict(io_workers=True, cpu_workers=2),
                dict(io_workers=4, cpu_workers=33)]:
        with pytest.raises(ValueError):
            widths(bad)
    for value in (0, 33, True, 1.0, '4'):
        with pytest.raises(ValueError):
            width(value)


def test_map_io_matches_serial_output_exactly():
    items = list(range(200))
    serial, serial_stats = map_io(identity_row, items, workers=1)
    parallel, parallel_stats = map_io(identity_row, items, workers=4, ahead=8)
    assert serial == parallel
    assert serial_stats['parallel'] is False
    assert parallel_stats['parallel'] is True
    assert parallel_stats['total'] == len(items)
    assert len(parallel_stats['task_seconds']) == len(items)
    assert 1 <= parallel_stats['peak_in_flight'] <= 8
    assert json.loads(json.dumps(parallel_stats))['width'] == 4


def test_map_io_preserves_order_not_completion_order():
    # Later items are slower, so completion order is the reverse of input order.
    items = list(range(60))
    parallel, _ = map_io(lambda value: (time.sleep((60 - value) / 20000), value)[1],
                         items, workers=4, ahead=8)
    assert parallel == items


def test_map_io_stops_at_first_failure():
    with pytest.raises(ValueError, match='injected failure'):
        map_io(fail_on, list(range(50)), workers=4, ahead=4)


def test_map_io_single_item_runs_serially():
    _, stats = map_io(identity_row, [1], workers=4)
    assert stats['parallel'] is False and stats['total'] == 1


def test_map_cpu_matches_serial_output_exactly():
    items = list(range(40))
    serial, serial_stats = map_cpu(identity_row, items, workers=1)
    parallel, parallel_stats = map_cpu(identity_row, items, workers=4, window=8,
                                       min_parallel=8)
    assert serial == parallel == [identity_row(v) for v in items]
    assert serial_stats['parallel'] is False
    assert parallel_stats['parallel'] is True
    assert parallel_stats['width'] == 4


def test_map_cpu_chunking_preserves_order_and_totals():
    items = list(range(45))
    chunked, stats = map_cpu(identity_row, items, workers=3, window=4, min_parallel=8,
                             chunk=8)
    assert chunked == [identity_row(v) for v in items]
    # One duration per dispatched chunk, with the unit named so it cannot be
    # mistaken for per-item timings; totals still count items.
    assert stats['unit'] == 'chunk' and len(stats['task_seconds']) == 6
    assert stats['total'] == len(items)
    serial, serial_stats = map_cpu(identity_row, items, workers=1)
    assert serial_stats['unit'] == 'item' and len(serial_stats['task_seconds']) == len(items)


def test_map_cpu_small_panel_stays_serial():
    _, stats = map_cpu(identity_row, list(range(8)), workers=4, min_parallel=16)
    assert stats['parallel'] is False


def test_map_cpu_rejects_unknown_thresholds():
    with pytest.raises(ValueError):
        map_cpu(double, [1, 2], workers=4, min_parallel=0)
    with pytest.raises(ValueError):
        map_cpu(double, [1, 2], workers=4, window=0)
    with pytest.raises(ValueError):
        map_cpu(double, [1, 2], workers=4, chunk=0)


def test_map_cpu_propagates_child_failure():
    with pytest.raises(ValueError, match='injected failure'):
        map_cpu(fail_on, list(range(40)), workers=4, window=4, min_parallel=8)


def test_stats_records_monotone_durations_and_derived_speedup():
    stats = Stats('io', 2, 3, True)
    stats.task(0.5)
    stats.task(0.25)
    stats.observe_in_flight(2)
    row = stats.to_dict()
    assert row['task_seconds'] == [0.5, 0.25]
    assert row['task_seconds_total'] == 0.75
    assert row['peak_in_flight'] == 2
    assert row['speedup'] > 0
    with pytest.raises(ValueError):
        stats.task(-1)
    with pytest.raises(ValueError):
        Stats('gpu', 1, 1, True)


def test_content_key_binds_complete_inputs():
    first = content_key({'config': 1}, ['a', 'b'], 'code-sha')
    assert first == content_key({'config': 1}, ['a', 'b'], 'code-sha')
    assert first != content_key({'config': 2}, ['a', 'b'], 'code-sha')
    assert first != content_key({'config': 1}, ['b', 'a'], 'code-sha')
    assert len(first) == 64
    with pytest.raises(ValueError):
        content_key()


def test_singleflight_reuses_identical_key_and_separates_others():
    flight = singleflight(max_entries=2)
    calls = []

    def compute():
        calls.append(1)
        return {'evidence': len(calls)}

    key = content_key('audit', 'v1')
    first, reused_first = flight.get(key, compute)
    second, reused_second = flight.get(key, compute)
    assert first == second == {'evidence': 1}
    assert (reused_first, reused_second) == (False, True)
    assert len(calls) == 1
    other, reused_other = flight.get(content_key('audit', 'v2'), compute)
    assert other == {'evidence': 2} and reused_other is False
    # Returned evidence is a copy, so a consumer cannot corrupt the cache.
    first['evidence'] = 99
    assert flight.get(key, compute)[0] == {'evidence': 1}
    with pytest.raises(ValueError):
        flight.get('short', compute)
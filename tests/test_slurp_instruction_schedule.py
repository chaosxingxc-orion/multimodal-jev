from collections import Counter
import pytest

from multimodal_jev.slurp_instruction_schedule import epoch_schedule, resume_cursor


def population():
    return [dict(id=f'a-{i}', source_group=f'g-{i}', intent='head' if i < 7 else 'tail',
                 role='learning', known_pending_source=False) for i in range(9)]


def test_full_coverage_padding_and_order_independence():
    rows = population()
    batches, report = epoch_schedule(rows, seed=317, epoch=0, global_batch=8)
    assert epoch_schedule(list(reversed(rows)), seed=317, epoch=0, global_batch=8) == (batches, report)
    real = [r for batch in batches for r in batch if not r['padding']]
    assert Counter(r['id'] for r in real) == Counter(r['id'] for r in rows)
    assert sum(r['weight'] for b in batches for r in b) == 9
    assert report['updates'] == 2 and report['padding_slots'] == 7
    assert set(r['id'] for r in batches[0][:4]) & {'a-7', 'a-8'}


def test_view_and_binding_do_not_depend_on_gold_label():
    rows = population()
    a, _ = epoch_schedule(rows, seed=317, epoch=1)
    b, _ = epoch_schedule([dict(r, intent='changed') for r in rows], seed=317, epoch=1)
    options = lambda batches: {r['id']: (r['view'], r['variant']) for b in batches for r in b if not r['padding']}
    assert options(a) == options(b)
    seen = {r['id']: set() for r in rows}
    for epoch in range(4):
        batches, _ = epoch_schedule(rows, seed=317, epoch=epoch)
        for r in batches[0]:
            seen[r['id']].add(r['view'])
    assert all(len(v) == 4 for v in seen.values())


def test_resume_rejects_changed_population_or_cursor():
    _, report = epoch_schedule(population(), seed=317, epoch=0, global_batch=8)
    assert resume_cursor(report, next_batch=2, actual_identity=report['schedule_sha256'])['complete']
    with pytest.raises(ValueError):
        resume_cursor(report, next_batch=3, actual_identity=report['schedule_sha256'])
    with pytest.raises(ValueError):
        resume_cursor(report, next_batch=0, actual_identity='wrong')
    with pytest.raises(ValueError):
        epoch_schedule([dict(population()[0], role='selection')], seed=317, epoch=0)

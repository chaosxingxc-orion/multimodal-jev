import json
import pytest

from tools.reports.context_rollout_efficiency_summary import summarize


def fixture_run(tmp_path):
    for step in (1, 2):
        for rank in range(4):
            p = tmp_path / f'rank-{rank}' / f'update-{step:04d}.json'
            p.parent.mkdir(exist_ok=True)
            p.write_text(json.dumps(dict(update=step, real_queries=128,
                total_seconds=10+rank, preparation_seconds=6, actor_seconds=3,
                rollout=dict(total_seconds=5, reference_seconds=2 if step==1 else 0,
                             reference_misses=128 if step==1 else 0))))
    return tmp_path


def test_global_exposure_and_nested_costs(tmp_path):
    report = summarize(fixture_run(tmp_path), 2, 1)
    assert report['epochs'][0]['real_query_exposures'] == 128
    assert report['epochs'][0]['mean_seconds']['total'] == 13
    assert report['epochs'][0]['mean_seconds']['residual'] == 1
    assert report['epochs'][1]['reference_misses'] == 0
    assert len(report['source_sha256']) == 8


def test_missing_rank_rejected(tmp_path):
    root = fixture_run(tmp_path)
    (root/'rank-3/update-0002.json').unlink()
    with pytest.raises(FileNotFoundError): summarize(root, 2, 1)


@pytest.mark.parametrize('change', [dict(update=9), dict(total_seconds=float('nan')),
    dict(preparation_seconds=11), dict(real_queries=127)])
def test_invalid_identity_timing_and_exposure(tmp_path, change):
    root = fixture_run(tmp_path); p=root/'rank-1/update-0001.json'
    d=json.loads(p.read_text()); d.update(change); p.write_text(json.dumps(d))
    with pytest.raises(ValueError): summarize(root, 2, 1)

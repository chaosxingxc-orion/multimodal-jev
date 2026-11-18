import json

import pytest

from multimodal_jev import context_replica_macro_rollout as macro
from test_context_macro_rollout import Client, rows


def pool(tmp_path, monkeypatch, clients, rollout_queries=32, max_queries=128):
    from multimodal_jev import local_checkpoint
    monkeypatch.setattr(macro, 'check_paths', lambda _: None)
    monkeypatch.setattr(local_checkpoint, 'check_paths', lambda _: None)
    return macro.MacroRolloutPool(clients, tmp_path,
        runtime_identity={'profile': 'fixture'}, rollout_queries=rollout_queries,
        max_queries=max_queries)



def test_single_receives_complete_macro_and_matches_dual(tmp_path, monkeypatch):
    one = Client()
    single = pool(tmp_path/'one', monkeypatch, [one], rollout_queries=128)
    dual = pool(tmp_path/'two', monkeypatch, [Client(), Client()])
    batch = rows(128)
    a = single.score_many(batch, audio_token_id=9)
    b = dual.score_many(batch, audio_token_id=9)
    assert a == b
    assert len(one.calls) == 1 and len(one.calls[0]) == 128
    assert sum(len(c.calls) for c in dual.clients) == 4
    assert single.last_receipt['completed_indices'] == list(range(128))
    for engine in (single, dual):
        receipt = engine.switch(snapshot='s', policy_id='p1', adapter='a')
        assert len(receipt['replica_receipts']) == len(engine.clients)
        assert engine.bundle['policy_id'] == 'p1'


@pytest.mark.parametrize('replicas', [1, 2])
@pytest.mark.parametrize('bad', ['short', 'order', 'policy', 'timeout'])
def test_failure_never_commits_partial_macro(tmp_path, monkeypatch, replicas, bad):
    clients = [Client() for _ in range(replicas)]
    clients[-1].bad = bad
    engine = pool(tmp_path, monkeypatch, clients, rollout_queries=2)
    with pytest.raises((ValueError, TimeoutError)):
        engine.score_many(rows(8), audio_token_id=9)
    assert engine.phase == 'failed'
    assert not list(tmp_path.glob('*/COMPLETE.json'))
    with pytest.raises(ValueError):
        engine.score_many(rows(8), audio_token_id=9)


@pytest.mark.parametrize('replicas', [0, 9, True, 1.0])
def test_invalid_replica_count(replicas):
    with pytest.raises(ValueError):
        macro.plan_rollouts([1, 4, 3], 2, replicas=replicas)


def test_single_tail_preserves_duplicate_exposures(tmp_path, monkeypatch):
    engine = pool(tmp_path, monkeypatch, [Client()], rollout_queries=2)
    result = engine.score_many(rows(1)*5, audio_token_id=9)
    assert len(result) == 5
    manifest = json.loads((tmp_path/'batch-00000000/INPUT.json').read_text())
    assert [j['indices'] for j in manifest['jobs']] == [[0, 1], [2, 3], [4]]
    assert {j['replica_id'] for j in manifest['jobs']} == {0}


@pytest.mark.parametrize('replicas', [2, 3, 4])
def test_declared_replica_count_covers_wide_macro(tmp_path, monkeypatch, replicas):
    clients = [Client() for _ in range(replicas)]
    engine = pool(tmp_path, monkeypatch, clients, rollout_queries=48, max_queries=384)
    result = engine.score_many(rows(384), audio_token_id=9)
    assert len(result) == 384
    assert sum(len(call) for c in clients for call in c.calls) == 384
    assert engine.last_receipt['completed_indices'] == list(range(384))


@pytest.mark.parametrize('max_queries', [0, 4097, True])
def test_pool_rejects_macro_batch_outside_declared_bound(max_queries):
    with pytest.raises(ValueError):
        macro.MacroRolloutPool([object()], '/tmp/journal', runtime_identity={'p': 1},
                               rollout_queries=32, max_queries=max_queries)


def test_pool_rejects_more_than_eight_replicas(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        pool(tmp_path, monkeypatch, [Client() for _ in range(9)])


def test_replicas_require_distinct_exchange_directories(tmp_path, monkeypatch):
    clients = [Client(), Client()]
    for client in clients:
        client.exchange = tmp_path / 'shared-exchange'
    with pytest.raises(ValueError, match='independent exchange'):
        pool(tmp_path / 'journal', monkeypatch, clients)

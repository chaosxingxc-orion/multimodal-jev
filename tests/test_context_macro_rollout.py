import copy
import json
import threading

import pytest
import torch

from multimodal_jev import context_macro_rollout as macro
from multimodal_jev.context_logits_wire import WIRE
from multimodal_jev.span_request_identity import tensor_identity


def rows(n=8):
    return [dict(inputs=dict(input_ids=torch.tensor([[i, 1]]),
                             attention_mask=torch.ones(1, 2, dtype=torch.long),
                             input_features=torch.zeros(1, 2, 3),
                             input_features_mask=torch.ones(1, 3, dtype=torch.long)),
                 positions=torch.tensor([[1]])) for i in range(n)]


class Client:
    def __init__(self, barrier=None, bad=None):
        self.bundle = dict(policy_id='p0', contract={'base_id': 'base'})
        self.runtime_identity = {'profile': 'fixture'}
        self.phase = 'ready'; self.calls = []; self.barrier = barrier; self.bad = bad

    def score_many(self, batch, *, audio_token_id):
        if self.barrier: self.barrier.wait(timeout=5)
        self.calls.append([int(r['inputs']['input_ids'][0, 0]) for r in batch])
        records = []
        for row in batch:
            identity = dict(policy_id=self.bundle['policy_id'], positions=row['positions'][0].tolist(),
                            inputs={k: tensor_identity(v) for k, v in sorted(row['inputs'].items())})
            records.append(dict(policy_id=self.bundle['policy_id'], request_identity=identity,
                                request_id=macro.digest(identity), prob=[0.5], logits=[0.0], output_protocol=WIRE))
        if self.bad == 'short': return records[:-1]
        if self.bad == 'order': return records[::-1]
        if self.bad == 'prob': records[0]['prob'] = [0.9]
        if self.bad == 'policy': self.bundle['policy_id'] = 'other'
        if self.bad == 'timeout': raise TimeoutError('uncertain execution')
        return records

    def switch(self, **spec):
        if self.bad == 'switch': raise ValueError('activation refused')
        self.bundle['policy_id'] = spec['policy_id']
        return dict(policy_id=spec['policy_id'], head_marker_exact=True, native_loaded_tensors=112)

    def close(self): self.phase = 'closed'


def pool(tmp_path, monkeypatch, clients):
    monkeypatch.setattr(macro, 'check_paths', lambda paths: None)
    # Storage admission has its own tests; these fixtures may live on tmpfs.
    # Exercise real atomic commits and fsync without imposing a host mount type.
    from multimodal_jev import local_checkpoint
    monkeypatch.setattr(local_checkpoint, 'check_paths', lambda paths: None)
    return macro.MacroRolloutPool(clients, tmp_path, runtime_identity={'profile': 'fixture'}, rollout_queries=2)


def test_concurrent_whole_rollouts_and_ordered_complete_delivery(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)
    clients = [Client(barrier), Client(barrier)]
    engine = pool(tmp_path, monkeypatch, clients)
    result = engine.score_macro('first', rows(), audio_token_id=9)
    assert [r['request_identity']['inputs']['input_ids'] for r in result] == [
        tensor_identity(r['inputs']['input_ids']) for r in rows()]
    assert clients[0].calls == [[0, 1], [4, 5]]
    assert clients[1].calls == [[2, 3], [6, 7]]
    receipt = json.loads((tmp_path/'first/COMPLETE.json').read_text())
    assert receipt['completed_indices'] == list(range(8))
    assert not receipt['actor_update_performed']
    with pytest.raises(FileExistsError): engine.score_macro('first', rows(), audio_token_id=9)
    assert len(clients[0].calls) == 2


@pytest.mark.parametrize('bad', ['short', 'order', 'prob', 'policy', 'timeout'])
def test_partial_or_corrupt_results_never_commit_or_retry(tmp_path, monkeypatch, bad):
    clients = [Client(), Client(bad=bad)]; engine = pool(tmp_path, monkeypatch, clients)
    with pytest.raises((ValueError, TimeoutError)):
        engine.score_macro('failed', rows(4), audio_token_id=9)
    assert engine.phase == 'failed'
    assert not (tmp_path/'failed/COMPLETE.json').exists()
    assert (tmp_path/'failed/FAILED.json').exists()
    calls = copy.deepcopy([c.calls for c in clients])
    with pytest.raises(ValueError): engine.score_macro('retry', rows(4), audio_token_id=9)
    assert calls == [c.calls for c in clients]


def test_activation_requires_both_replicas(tmp_path, monkeypatch):
    clients = [Client(), Client()]; engine = pool(tmp_path, monkeypatch, clients)
    result = engine.switch(snapshot='s', policy_id='p1', adapter='a')
    assert len(result['replica_receipts']) == 2
    assert engine.bundle['policy_id'] == 'p1'
    clients[1].bad = 'switch'
    with pytest.raises(ValueError): engine.switch(snapshot='s2', policy_id='p2', adapter='a2')
    assert engine.phase == 'failed'
    with pytest.raises(ValueError): engine.score_many(rows(4), audio_token_id=9)


def test_stale_policy_and_label_leak_refused_before_dispatch(tmp_path, monkeypatch):
    clients = [Client(), Client()]; engine = pool(tmp_path, monkeypatch, clients)
    clients[1].bundle['policy_id'] = 'stale'
    with pytest.raises(ValueError): engine.score_many(rows(4), audio_token_id=9)
    clients[1].bundle['policy_id'] = 'p0'
    batch = rows(4); batch[0]['labels'] = [1]
    with pytest.raises(ValueError): engine.score_many(batch, audio_token_id=9)
    assert not any(c.calls for c in clients)


def test_length_schedule_preserves_all_queries_and_ownership():
    jobs = macro.plan_rollouts([1, 20, 5, 2, 16, 3, 7], 2)
    assert sorted(i for j in jobs for i in j['indices']) == list(range(7))
    assert len({j['rollout_id'] for j in jobs}) == len(jobs)
    assert {j['replica_id'] for j in jobs} == {0, 1}


def test_duplicate_input_exposures_preserve_distinct_schedule_slots(tmp_path, monkeypatch):
    engine = pool(tmp_path, monkeypatch, [Client(), Client()])
    batch = rows(1) * 4
    result = engine.score_macro('repeat', batch, audio_token_id=9)
    assert len(result) == 4
    assert engine.last_receipt['completed_indices'] == [0, 1, 2, 3]

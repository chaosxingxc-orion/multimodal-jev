"""Two real spawned IPC workers; model outputs and weight activation are fixtures."""
import copy
import json
import multiprocessing
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import uuid

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools/jev'))
from test_context_rollout_session import REF, P0, P1, BASE, spec
from test_context_dual_session_candidate import Engine, RUNTIME, query
from tools.reports.context_dual_session_candidate import RuntimeReferenceCache, DualContextRolloutSession
from tools.reports.context_batched_service_candidate import serve_connection
from tools.reports.context_batched_request_candidate import prepare_request
from tools.reports.context_macro_client import load_pool, connect_pool
from multimodal_jev.span_rollout_rpc import send

CONTRACT = {'base_id': BASE, 'token': {'id': 9}}
ALGORITHM = 'dual_credit_grpo_floor_clipped_v1'


def worker(address, exchange, channel, barrier):
    class Model:
        bundle = dict(policy_id=P0, contract=CONTRACT)
        max_model_len = 32
        max_decisions = 2
        max_batch_queries = 8
        max_batch_tokens = 64

        def __init__(self):
            self.events = []; self.seen = set()

        def score_many(self, rows, *, audio_token_id):
            pid = self.bundle['policy_id']
            if pid not in self.seen:
                barrier.wait(timeout=20); self.seen.add(pid)
            self.events.append(dict(operation='encode', policy=pid, size=len(rows)))
            z = torch.tensor({REF: [.5, -.5], P0: [-.5, .5], P1: [1., -1.]}[pid])
            records = []
            for row in rows:
                p = prepare_request(self, row, audio_token_id, SimpleNamespace)
                records.append(dict(policy_id=pid, request_id=p['request_id'],
                    request_identity=p['request_identity'], prob=z.sigmoid().tolist(),
                    logits=z.tolist(), output_protocol=RUNTIME['output_protocol']))
            return records

        def switch(self, **kw):
            self.bundle = dict(self.bundle, policy_id=kw['policy_id'])
            self.events.append(dict(operation='switch', policy=kw['policy_id']))
            return dict(policy_id=kw['policy_id'], head_marker_exact=True, native_loaded_tensors=112)

    model = Model()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(address); listener.listen(1); channel.send('ready')
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(30)
            send(connection, dict(protocol='dual-runtime-hello-v1', runtime_identity=RUNTIME,
                policy_id=P0, contract=CONTRACT))
            serve_connection(connection, model, exchange)
    channel.send(model.events); channel.close()


@pytest.fixture
def storage_isolation(monkeypatch):
    import context_reference_cache
    import context_policy_activation
    from multimodal_jev import local_runtime_storage, local_checkpoint, context_macro_rollout
    import tools.reports.context_macro_client as client
    for module in (context_reference_cache, local_runtime_storage, local_checkpoint,
                   context_macro_rollout, client):
        monkeypatch.setattr(module, 'check_paths', lambda _: None)
    monkeypatch.setattr(context_policy_activation, 'prepare_switch',
        lambda current, **kw: dict(current, policy_id=kw['policy_id']))


def session(engine, cache):
    return DualContextRolloutSession(engine, cache, runtime_identity=RUNTIME,
        candidate_ids=['b', 'a'], algorithm=ALGORITHM, reference=spec(REF),
        initial_policy=spec(P0), audio_token_id=1)


def cache_at(path):
    return RuntimeReferenceCache(path, policy_id=REF, base_id=BASE,
        contract=CONTRACT, runtime_identity=RUNTIME)


def equal_credit(expected, actual):
    left, lo, lr = expected; right, ro, rr = actual
    assert lo == ro and ro['algorithm'] == ALGORITHM
    assert lr['queries'] == rr['queries']
    assert len(left) == len(right) == 128
    for a, b in zip(left, right, strict=True):
        assert a['dual_credit']['binding_sha256'] == b['dual_credit']['binding_sha256']
        assert torch.equal(a['reference_probability'], b['reference_probability'])
        for branch in ('ranking', 'existence'):
            for field in ('actions', 'advantages', 'old_logprob', 'rewards'):
                assert torch.equal(a['dual_credit']['branches'][branch][field],
                                   b['dual_credit']['branches'][branch][field])


def test_two_process_macro128_grpo_reference_and_policy_barrier(tmp_path, storage_isolation):
    ctx = multiprocessing.get_context('spawn'); barrier = ctx.Barrier(2)
    workers = []; replicas = []; pool = None
    try:
        for i in range(2):
            address = '\0jev-macro-' + uuid.uuid4().hex
            exchange = str(tmp_path / f'exchange-{i}')
            parent, child = ctx.Pipe(duplex=False)
            process = ctx.Process(target=worker, args=(address, exchange, child, barrier))
            process.start(); child.close(); workers.append((process, parent))
            replicas.append(dict(socket=address, exchange=exchange))
        for _, pipe in workers: assert pipe.poll(30) and pipe.recv() == 'ready'
        macros = dict(MACRO_ATTEMPT_ID='attempt-1', MACRO_JOURNAL_ROOT=str(tmp_path/'journal'),
            DUAL_RUNTIME_IDENTITY=RUNTIME, ROLLOUT_0_SOCKET=replicas[0]['socket'],
            ROLLOUT_0_EXCHANGE=replicas[0]['exchange'], ROLLOUT_1_SOCKET=replicas[1]['socket'],
            ROLLOUT_1_EXCHANGE=replicas[1]['exchange'])
        local = tmp_path/'local.json'
        local.write_text(json.dumps(dict(schema='jev-local-macros-v1', macros=macros)))
        template = Path(__file__).parents[1]/'configs/portable/operations/context-macro-client.json'
        pool, provenance = load_pool(template, local, bundle=dict(policy_id=P0, contract=CONTRACT))
        assert len(provenance['macros_used']) == 7
        cache = cache_at(tmp_path/'cache'); remote = session(pool, cache)
        oracle_cache = cache_at(tmp_path/'oracle'); oracle_engine = Engine(oracle_cache)
        oracle_engine.bundle['contract'] = CONTRACT
        oracle = session(oracle_engine, oracle_cache)
        items = [query(P0, i) for i in range(127)]
        for row in items:
            row['inputs'].update(input_ids=torch.tensor([[1, 9, 9]]),
                attention_mask=torch.ones(1, 3, dtype=torch.long),
                input_features_mask=torch.ones(1, 3, dtype=torch.long))
        pad = copy.deepcopy(items[0]); pad.update(request_id='padding', real_request=False, request_weight=0.)
        items.append(pad)
        a = oracle.prepare(items, update=0); b = remote.prepare(items, update=0)
        equal_credit(a, b); assert b[2]['reference_misses'] == 127
        with pytest.raises(ValueError, match='ready'): remote.prepare(items, update=0)
        for s in (oracle, remote): s.accept_actor_update(update=0, next_policy=spec(P1))
        assert all(c.bundle['policy_id'] == P1 for c in pool.clients)
        for row in items: row['policy_id'] = P1
        a = oracle.prepare(items, update=1); b = remote.prepare(items, update=1)
        equal_credit(a, b); assert b[2]['reference_misses'] == 0
        assert pool.sequence == 3 and len(list(cache.namespace.glob('*.json'))) == 127
        for index, count in enumerate((127, 128, 128)):
            directory = pool.root/f'batch-{index:08d}'
            manifest = json.loads((directory/'INPUT.json').read_text())
            receipt = json.loads((directory/'COMPLETE.json').read_text())
            assert receipt['completed_indices'] == list(range(count))
            assert {j['replica_id'] for j in manifest['jobs']} == {0, 1}
            assert sorted(i for j in manifest['jobs'] for i in j['indices']) == list(range(count))
            assert not receipt['actor_update_performed']
        assert all(not list(Path(r['exchange']).glob('*.pt')) for r in replicas)
        pool.close(); pool = None
        for _, pipe in workers:
            assert pipe.poll(30); events = pipe.recv()
            assert [e['policy'] for e in events if e['operation'] == 'switch'] == [REF, P0, P1]
            encodes = [e for e in events if e['operation'] == 'encode']
            assert len(encodes) == 24 and all(1 <= e['size'] <= 8 for e in encodes)
    finally:
        if pool is not None: pool.close()
        for process, pipe in workers:
            pipe.close(); process.join(30)
            if process.is_alive(): process.terminate(); process.join(5)
        assert all(p.exitcode == 0 for p, _ in workers)


@pytest.mark.parametrize('fault', ['same_socket', 'same_exchange', 'existing_attempt'])
def test_invalid_connection_configuration_before_socket(tmp_path, storage_isolation, fault):
    config = dict(schema='jev-macro-client-v1', attempt_id='attempt', journal_root=str(tmp_path),
        runtime_identity=RUNTIME, replicas=[dict(socket='one', exchange=str(tmp_path/'a')),
        dict(socket='two', exchange=str(tmp_path/'b'))], timeout_seconds=1, rollout_queries=32, max_queries=128)
    if fault == 'same_socket': config['replicas'][1]['socket'] = 'one'
    if fault == 'same_exchange': config['replicas'][1]['exchange'] = str(tmp_path/'a')
    if fault == 'existing_attempt': (tmp_path/'attempt').mkdir()
    with pytest.raises((ValueError, FileExistsError)):
        connect_pool(config, bundle=dict(policy_id=P0, contract=CONTRACT))

import copy
import socket
import threading
from types import SimpleNamespace as NS

import pytest
import torch

from multimodal_jev.context_rollout_batch import bind_rollout
from tools.reports.context_batched_service_candidate import plan_batches, score_rows, serve_connection
from tools.reports.context_bulk_rpc_candidate import BulkContextEngineClient
from test_context_batched_request_candidate import case
from test_context_bulk_rpc_candidate import record


def engine_case():
    owner, pair, _ = case()
    owner.bundle['contract']['base_id'] = 'b'*64
    calls = []
    def score_many(rows, *, audio_token_id):
        calls.append([r['inputs']['input_ids'].shape[1] for r in rows])
        return [record(owner.bundle['policy_id'], r) for r in rows]
    owner.score_many = score_many
    return owner, pair, calls


def test_dedup_restores_original_rows_and_sampling_without_cross_command_cache():
    owner, pair, calls = engine_case()
    rows = [pair[0], pair[1], copy.deepcopy(pair[0])]
    got, receipt = score_rows(owner, rows, policy_id='a'*64, audio_token_id=98)
    assert calls == [[7, 7]] and receipt['rows'] == 3 and receipt['unique_rows'] == 2
    assert got == [record('a'*64, r) for r in rows]
    assert got[0] is not got[2]
    for i, row in enumerate(rows):
        reference = bind_rollout(record('a'*64, row), **row, policy_id='a'*64, group_size=16, seed=317+i)
        actual = bind_rollout(got[i], **row, policy_id='a'*64, group_size=16, seed=317+i)
        assert torch.equal(actual['actions'], reference['actions'])
    score_rows(owner, rows, policy_id='a'*64, audio_token_id=98)
    assert calls == [[7, 7], [7, 7]]


def length_row(row, length):
    result = copy.deepcopy(row)
    extra = length - 7
    result['inputs']['input_ids'] = torch.cat([torch.ones(1, extra, dtype=torch.long),
                                              result['inputs']['input_ids']], dim=1)
    result['inputs']['attention_mask'] = torch.ones(1, length, dtype=torch.long)
    result['positions'] += extra
    return result


def test_length_packing_respects_both_budgets_and_restores_order():
    owner, pair, calls = engine_case()
    owner.max_batch_queries = 2; owner.max_batch_tokens = 20
    rows = [length_row(pair[0], n) for n in [11, 7, 12, 8, 9]]
    got, receipt = score_rows(owner, rows, policy_id='a'*64, audio_token_id=98)
    assert calls == [[7, 8], [9, 11], [12]]
    assert receipt['batch_tokens'] == [15, 20, 12]
    assert got == [record('a'*64, r) for r in rows]


@pytest.mark.parametrize('fault', ['last_invalid', 'too_long', 'stale', 'bad_capacity'])
def test_command_prevalidated_before_first_model_call(fault):
    owner, rows, calls = engine_case(); policy='a'*64
    if fault == 'last_invalid': rows[-1]['positions'][0, 0] = 0
    if fault == 'too_long': owner.max_batch_tokens = 6
    if fault == 'stale': policy='c'*64
    if fault == 'bad_capacity': owner.max_batch_queries=True
    with pytest.raises(ValueError): score_rows(owner, rows, policy_id=policy, audio_token_id=98)
    assert not calls


@pytest.mark.parametrize('fault', ['missing', 'reorder', 'changed_policy', 'wrong_probability'])
def test_model_failure_does_not_return_partial_command(fault):
    owner, rows, calls = engine_case(); original=owner.score_many
    def score(*args, **kwargs):
        result=original(*args, **kwargs)
        if fault=='missing':result.pop()
        if fault=='reorder':result.reverse()
        if fault=='changed_policy':owner.bundle=dict(owner.bundle,policy_id='c'*64)
        if fault=='wrong_probability':result[-1]['prob'][0]=float('nan')
        return result
    owner.score_many=score
    with pytest.raises(ValueError):score_rows(owner,rows,policy_id='a'*64,audio_token_id=98)
    assert len(calls)==1


@pytest.mark.parametrize('fail', [False, True])
def test_existing_bulk_client_actual_socket_and_policy_switch(tmp_path, monkeypatch, fail):
    import multimodal_jev.local_runtime_storage as storage
    import multimodal_jev.local_checkpoint as checkpoint
    import context_policy_activation as activation
    monkeypatch.setattr(storage, 'check_paths', lambda p: None)
    monkeypatch.setattr(checkpoint, 'check_paths', lambda p: None)
    monkeypatch.setattr(activation, 'prepare_switch', lambda current, **kw: dict(current, policy_id=kw['policy_id']))
    engine, pair, calls=engine_case(); errors=[]
    rows=[pair[0],pair[1],copy.deepcopy(pair[0])]
    if fail: rows[-1]['positions'][0,0]=0
    def switch(**kw):
        engine.bundle=dict(engine.bundle,policy_id=kw['policy_id'])
        return dict(policy_id=kw['policy_id'],head_marker_exact=True,native_loaded_tensors=112)
    engine.switch=switch
    address=tmp_path/'worker.sock'
    listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);listener.bind(str(address));listener.listen(1)
    def worker():
        try:
            conn,_=listener.accept()
            with conn:
                conn.settimeout(5);serve_connection(conn,engine,tmp_path)
        except BaseException as exc:errors.append(exc)
        finally:listener.close()
    thread=threading.Thread(target=worker);thread.start()
    client=BulkContextEngineClient(address,tmp_path,bundle=copy.deepcopy(engine.bundle),timeout=5)
    try:
        if fail:
            with pytest.raises(ValueError):client.score_many(rows,audio_token_id=98)
            assert client.phase=='failed' and not calls
            with pytest.raises(ValueError):client.score_many(rows,audio_token_id=98)
            assert list(tmp_path.glob('bulk-*.pt'))
        else:
            assert client.score_many(rows,audio_token_id=98)==[record('a'*64,r) for r in rows]
            assert calls==[[7,7]]
            client.switch(snapshot='snapshot',policy_id='c'*64,adapter='adapter')
            assert client.score_many(rows,audio_token_id=98)==[record('c'*64,r) for r in rows]
            assert calls==[[7,7],[7,7]]
            assert not list(tmp_path.glob('bulk-*.pt'))
    finally:
        client.close();thread.join(5)
    assert not thread.is_alive()
    assert bool(errors)==fail

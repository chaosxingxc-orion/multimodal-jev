import copy
import socket
import threading

import pytest
import torch

from multimodal_jev.context_logits_wire import WIRE
from tools.reports.context_replica_batched_service import serve_connection
from tools.reports.context_replica_macro_client import connect_pool
from tools.reports.context_replica_runtime import hello
from multimodal_jev.span_rollout_rpc import send
from test_context_batched_service_candidate import engine_case
from test_context_bulk_rpc_candidate import record


@pytest.mark.parametrize('replicas,batch', [(1,32),(1,48),(2,32),(2,48)])
def test_complete_macro192_real_sockets_and_internal_batches(tmp_path,monkeypatch,replicas,batch):
    run_case(tmp_path,monkeypatch,replicas,batch)


@pytest.mark.parametrize('fault',['last_invalid','middle_failure','wrong_hello'])
def test_macro192_failure_never_delivers_partial(tmp_path,monkeypatch,fault):
    run_case(tmp_path,monkeypatch,1,48,fault)


def run_case(tmp_path,monkeypatch,replicas,batch,fault=None):
    from multimodal_jev import local_checkpoint,local_runtime_storage,context_replica_macro_rollout
    from tools.reports import context_replica_macro_client
    import context_policy_activation
    for m in (local_checkpoint,local_runtime_storage,context_replica_macro_rollout,context_replica_macro_client):
        monkeypatch.setattr(m,'check_paths',lambda _:None)
    monkeypatch.setattr(context_policy_activation,'prepare_switch',
                        lambda current,**kw:dict(current,policy_id=kw['policy_id']))
    runtime={'fixture':'macro192'};endpoints=[];threads=[];errors=[];calls=[]
    owner,pair,_=engine_case(); bundle=copy.deepcopy(owner.bundle)
    rows=[]
    for i in range(192):
        row=copy.deepcopy(pair[0]);row['inputs']['input_features'][0,0,0]=i/192
        rows.append(row)
    if fault=='last_invalid':rows[-1]['positions'][0,0]=0
    for index in range(replicas):
        root=tmp_path/str(index);root.mkdir()
        engine,_,_=engine_case();engine.max_batch_queries=batch;engine.max_batch_tokens=10000
        counted=[];calls.append(counted)
        def score(part,*,audio_token_id,engine=engine,counted=counted):
            counted.append(len(part))
            if fault=='middle_failure' and len(counted)==2:raise ValueError('fixture failure')
            result=[record(engine.bundle['policy_id'],r) for r in part]
            z=torch.logit(torch.tensor([.3,.7])).tolist()
            for value in result:value.update(logits=z,output_protocol=WIRE)
            return result
        engine.score_many=score
        def switch(engine=engine,**kw):
            engine.bundle=dict(engine.bundle,policy_id=kw['policy_id'])
            return dict(policy_id=kw['policy_id'],head_marker_exact=True,native_loaded_tensors=112)
        engine.switch=switch
        address=root/'rpc.sock';listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        listener.bind(str(address));listener.listen(1);listener.settimeout(5)
        endpoints.append(dict(socket=str(address),exchange=str(root)))
        def worker(listener=listener,engine=engine,root=root):
            try:
                conn,_=listener.accept()
                with conn:
                    conn.settimeout(5)
                    handshake=hello(runtime,engine.bundle)
                    if fault=='wrong_hello':handshake['max_request_queries']=128
                    send(conn,handshake);serve_connection(conn,engine,root)
            except BaseException as exc:errors.append(exc)
            finally:listener.close()
        t=threading.Thread(target=worker);t.start();threads.append(t)
    config=dict(schema='jev-replica-macro-client-v1',attempt_id='test',journal_root=str(tmp_path/'journal'),
        runtime_identity=runtime,replicas=endpoints,timeout_seconds=5,
        rollout_queries=192//replicas,max_queries=192)
    pool=None
    try:
        if fault=='wrong_hello':
            with pytest.raises(ValueError,match='capacity'):connect_pool(config,bundle=bundle)
            assert not any(calls)
        else:
            pool=connect_pool(config,bundle=bundle)
            if fault:
                with pytest.raises(ValueError):pool.score_many(rows,audio_token_id=98)
                assert pool.phase=='failed'
                with pytest.raises(ValueError):pool.score_many(rows,audio_token_id=98)
                assert not list((tmp_path/'journal').glob('*/*/COMPLETE.json'))
                if fault=='last_invalid':assert not any(calls)
            else:
                result=pool.score_many(rows,audio_token_id=98)
                assert len(result)==192 and pool.last_receipt['completed_indices']==list(range(192))
                assert [v['request_id'] for v in result]==[record(bundle['policy_id'],r)['request_id'] for r in rows]
                assert calls==[[batch]*(192//replicas//batch) for _ in range(replicas)]
                assert len(pool.switch(snapshot='s',policy_id='c'*64,adapter='a')['replica_receipts'])==replicas
    finally:
        if pool:pool.close()
        for t in threads:t.join(6)
    assert not any(t.is_alive() for t in threads)
    if fault is None:assert not errors

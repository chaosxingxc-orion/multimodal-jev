import hashlib
import json
import socket
import threading
import pytest
import torch
from tools.reports import context_bulk_rpc_candidate as m
from multimodal_jev.span_request_identity import tensor_identity
from multimodal_jev.context_rollout_batch import bind_rollout


def row(n):
    return dict(inputs=dict(input_ids=torch.arange(n).reshape(1,n),attention_mask=torch.ones(1,n,dtype=torch.long),
        input_features=torch.ones(1,2,n),input_features_mask=torch.ones(1,n,dtype=torch.long)),positions=torch.tensor([[0,n-1]]))


def record(policy,value):
    identity=dict(policy_id=policy,positions=value['positions'][0].tolist(),inputs={k:tensor_identity(v) for k,v in sorted(value['inputs'].items())})
    return dict(policy_id=policy,request_identity=identity,request_id=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest(),prob=[.3,.7])


def connect(tmp_path,monkeypatch,*,fail=False):
    import multimodal_jev.local_runtime_storage as storage
    import multimodal_jev.local_checkpoint as checkpoint
    import context_policy_activation as activation
    monkeypatch.setattr(storage,'check_paths',lambda p:None)
    monkeypatch.setattr(checkpoint,'check_paths',lambda p:None)
    monkeypatch.setattr(activation,'prepare_switch',lambda current,**kw:dict(current,policy_id=kw['policy_id']))
    address=tmp_path/'worker.sock';listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);listener.bind(str(address));listener.listen(1)
    calls=[];errors=[]
    class Engine:
        bundle={'policy_id':'a'*64,'contract':{'base_id':'b'*64}}
        def score(self,inputs,positions,*,audio_token_id):
            calls.append((self.bundle['policy_id'],inputs['input_ids'].shape[1]))
            if fail and len(calls)==2:raise RuntimeError('Mid-batch failure')
            return record(self.bundle['policy_id'],dict(inputs=inputs,positions=positions))
        def switch(self,**kw):
            self.bundle=dict(self.bundle,policy_id=kw['policy_id'])
            return dict(policy_id=kw['policy_id'],head_marker_exact=True,native_loaded_tensors=112)
    def worker():
        try:
            conn,_=listener.accept()
            with conn:conn.settimeout(5);m.serve_connection(conn,Engine(),tmp_path)
        except BaseException as exc:errors.append(exc)
        finally:listener.close()
    t=threading.Thread(target=worker);t.start()
    client=m.BulkContextEngineClient(address,tmp_path,bundle=dict(Engine.bundle),timeout=5)
    return client,t,calls,errors


@pytest.mark.parametrize('count',[3,128])
def test_actual_socket_bulk_equals_sequential_and_switch(tmp_path,monkeypatch,count):
    client,t,calls,errors=connect(tmp_path,monkeypatch);lengths=([3,7,3]*43)[:count];rows=[row(n) for n in lengths]
    try:
        result=client.score_many(rows,audio_token_id=1)
        assert client.sequence==1
        assert result==[record('a'*64,r) for r in rows]
        for i,(r,got) in enumerate(zip(rows,result)):
            expected=bind_rollout(record('a'*64,r),**r,policy_id='a'*64,group_size=16,seed=i)
            actual=bind_rollout(got,**r,policy_id='a'*64,group_size=16,seed=i)
            assert torch.equal(actual['actions'],expected['actions'])
        client.switch(snapshot='s',policy_id='c'*64,adapter='a')
        assert client.score(**row(5),audio_token_id=1)['policy_id']=='c'*64
    finally:client.close();t.join(5)
    assert not errors and not t.is_alive() and not list(tmp_path.glob('bulk-*.pt'))
    assert calls==[('a'*64,n) for n in lengths]+[('c'*64,5)]


def test_partial_failure_preserves_payload_and_forbids_retry(tmp_path,monkeypatch):
    client,t,calls,errors=connect(tmp_path,monkeypatch,fail=True)
    try:
        with pytest.raises(ValueError):client.score_many([row(3),row(7),row(9)],audio_token_id=1)
        assert client.phase=='failed'
        with pytest.raises(ValueError):client.score_many([row(3)],audio_token_id=1)
        assert list(tmp_path.glob('bulk-*.pt'))
    finally:client.close();t.join(5)
    assert len(calls)==2 and len(errors)==1 and not t.is_alive()


@pytest.mark.parametrize('fault',['hash','outside','labels','nested_labels','empty','oversize'])
def test_read_payload_rejects_invalid_batch(tmp_path,fault):
    rows=[row(3)];p=tmp_path/'x.pt';exchange=tmp_path
    if fault=='labels':rows[0]['labels']=[True,False]
    if fault=='nested_labels':rows[0]['inputs']['labels']=torch.ones(1)
    if fault=='empty':rows=[]
    if fault=='oversize':rows=rows*129
    torch.save(rows,p);digest=hashlib.sha256(p.read_bytes()).hexdigest()
    if fault=='hash':digest='0'*64
    if fault=='outside':exchange=tmp_path/'another'
    with pytest.raises(ValueError):m.read_rows(p,digest,exchange)

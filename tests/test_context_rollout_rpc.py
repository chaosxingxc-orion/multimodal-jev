"""Actual Unix socket/file exchange; substitute engine, no NPU claim."""
import hashlib,json,socket,sys,threading
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import context_rollout_rpc as m
from multimodal_jev.span_request_identity import tensor_identity


def payload():
    return dict(inputs=dict(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.ones(1,3,dtype=torch.long),
        input_features=torch.ones(1,2,3),input_features_mask=torch.ones(1,3,dtype=torch.long)),positions=torch.tensor([[1,2]]))


def record(policy,value):
    ident=dict(policy_id=policy,positions=value['positions'][0].tolist(),
        inputs={k:tensor_identity(v) for k,v in sorted(value['inputs'].items())})
    return dict(policy_id=policy,request_identity=ident,
        request_id=hashlib.sha256(json.dumps(ident,sort_keys=True,separators=(',',':')).encode()).hexdigest(),prob=[.3,.7])


@pytest.mark.parametrize('fault',['hash','outside','label','input_label'])
def test_payload_rejection(tmp_path,fault):
    value=payload();p=tmp_path/'x.pt';exchange=tmp_path
    if fault=='label':value['labels']=torch.tensor([1,0])
    if fault=='input_label':value['inputs']['labels']=torch.tensor([1,0])
    torch.save(value,p);digest=hashlib.sha256(p.read_bytes()).hexdigest()
    if fault=='hash':digest='0'*64
    if fault=='outside':exchange=tmp_path/'other'
    with pytest.raises(ValueError):m.read_inputs(p,digest,exchange)


def test_client_server_score_switch_score(tmp_path,monkeypatch):
    import multimodal_jev.local_runtime_storage as storage
    import multimodal_jev.local_checkpoint as checkpoint
    import context_policy_activation as activation
    monkeypatch.setattr(storage,'check_paths',lambda paths:None)
    monkeypatch.setattr(checkpoint,'check_paths',lambda paths:None)
    monkeypatch.setattr(activation,'prepare_switch',lambda current,**k:dict(current,policy_id=k['policy_id']))
    address=tmp_path/'engine.sock';listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    listener.bind(str(address));listener.listen(1);errors=[];calls=[]
    bundle=dict(policy_id='a'*64,contract={'base_id':'b'*64})
    class Engine:
        def __init__(self):self.bundle=dict(bundle)
        def score(self,inputs,positions,*,audio_token_id):
            calls.append(('score',self.bundle['policy_id']))
            return record(self.bundle['policy_id'],dict(inputs=inputs,positions=positions))
        def switch(self,**k):
            calls.append(('switch',k['policy_id']));self.bundle['policy_id']=k['policy_id']
            return dict(policy_id=k['policy_id'],head_marker_exact=True,native_loaded_tensors=112)
    def worker():
        try:
            conn,_=listener.accept()
            with conn:conn.settimeout(5);m.serve_connection(conn,Engine(),tmp_path)
        except BaseException as exc:errors.append(exc)
        finally:listener.close()
    thread=threading.Thread(target=worker);thread.start()
    client=m.ContextEngineClient(address,tmp_path,bundle=bundle,timeout=5)
    try:
        assert client.score(**payload(),audio_token_id=1)['policy_id']=='a'*64
        client.switch(snapshot='snapshot',policy_id='c'*64,adapter='adapter')
        assert client.score(**payload(),audio_token_id=1)['policy_id']=='c'*64
    finally:client.close();thread.join(5)
    assert not thread.is_alive() and not errors and not list(tmp_path.glob('inputs-*.pt'))
    assert calls==[('score','a'*64),('switch','c'*64),('score','c'*64)]


@pytest.mark.parametrize('fault',['stale','sequence','label'])
def test_server_rejects_before_engine_execution(tmp_path,fault):
    value=payload();p=tmp_path/'x.pt';torch.save(value,p)
    command=dict(sequence=0,operation='score',policy_id='a',path=str(p),
        sha256=hashlib.sha256(p.read_bytes()).hexdigest(),audio_token_id=1)
    if fault=='stale':command['policy_id']='wrong'
    if fault=='sequence':command['sequence']=1
    if fault=='label':command['labels']=[1,0]
    left,right=socket.socketpair();errors=[]
    class Engine:
        bundle={'policy_id':'a'}
        def score(self,*args,**kw):raise AssertionError('must not call engine')
    def worker():
        try:
            with right:m.serve_connection(right,Engine(),tmp_path)
        except BaseException as exc:errors.append(exc)
    t=threading.Thread(target=worker);t.start()
    with left:
        left.settimeout(5);m.send(left,command);response=m.receive(left)
    t.join(5)
    assert not response['ok'] and len(errors)==1 and isinstance(errors[0],ValueError)

"""Real local socket framing and tensor transport; engine is a substitute."""
import hashlib
import socket
import struct
import threading
import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.span_rollout_rpc import send, receive, read_renderer, serve_connection


def payload(tmp_path):
    path=tmp_path/'renderer.pt'
    rows=[dict(id='audio',inputs={'input_ids':torch.tensor([[1,2]])},positions=torch.tensor([[1]]),
        candidates=[[0,1]],request_id='request',identity={'policy_id':'base:adapter'})]
    torch.save(rows,path)
    return path,hashlib.sha256(path.read_bytes()).hexdigest(),rows


def test_actual_socket_stream_and_weights_only_payload(tmp_path):
    path,digest,rows=payload(tmp_path);left,right=socket.socketpair();calls=[];errors=[]
    class Engine:
        def rollout(self,config,renderer):
            assert torch.equal(renderer[0][1]['input_ids'],rows[0]['inputs']['input_ids'])
            assert renderer[0][-1]=='processor'
            calls.append(config['adapter_id'])
            return dict(passed=True,adapter_id=config['adapter_id'])
    def worker():
        try:
            with right:serve_connection(right,Engine(),tmp_path,'processor')
        except BaseException as exc:errors.append(exc)
    thread=threading.Thread(target=worker);thread.start()
    with left:
        left.settimeout(5)
        for index,adapter in enumerate(('old','new','old')):
            send(left,dict(sequence=index,operation='rollout',config={'adapter_id':adapter},
                renderer=str(path),sha256=digest))
            assert receive(left)==dict(sequence=index,ok=True,result=dict(passed=True,adapter_id=adapter))
    thread.join(5)
    assert not thread.is_alive() and not errors and calls==['old','new','old']


@pytest.mark.parametrize('fault',['hash','outside','label','input_label'])
def test_reject_changed_or_nonrenderer_payload(tmp_path,fault):
    path,digest,rows=payload(tmp_path);exchange=tmp_path
    if fault=='hash':digest='0'*64
    elif fault=='outside':exchange=tmp_path/'other'
    else:
        if fault=='label':rows[0]['targets']=torch.tensor([1])
        else:rows[0]['inputs']['labels']=torch.tensor([1])
        torch.save(rows,path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):read_renderer(path,digest,exchange,None)


def test_partial_message_and_oversize_are_rejected():
    for raw,exception in [(struct.pack('!I',10)+b'{}',ConnectionError),(b'\x00',ConnectionError),
                          (struct.pack('!I',2**30),ValueError)]:
        left,right=socket.socketpair()
        with left,right:
            left.sendall(raw);left.shutdown(socket.SHUT_WR)
            with pytest.raises(exception):receive(right)


def test_repeated_sequence_fails_without_second_engine_call(tmp_path):
    path,digest,_=payload(tmp_path);left,right=socket.socketpair();calls=[];errors=[]
    class Engine:
        def rollout(self,*args):calls.append(1);return {'passed':True}
    def worker():
        try:
            with right:serve_connection(right,Engine(),tmp_path,None)
        except ValueError as exc:errors.append(str(exc))
    thread=threading.Thread(target=worker);thread.start()
    with left:
        left.settimeout(5)
        command=dict(sequence=0,operation='rollout',config={},renderer=str(path),sha256=digest)
        send(left,command);assert receive(left)['ok']
        send(left,command);assert receive(left)['ok'] is False
    thread.join(5)
    assert not thread.is_alive() and calls==[1] and errors


def test_partial_request_is_failure_not_clean_service_completion(tmp_path):
    left,right=socket.socketpair()
    with left,right:
        left.sendall(struct.pack('!I',100)+b'{')
        left.shutdown(socket.SHUT_WR)
        with pytest.raises(ConnectionError):serve_connection(right,None,tmp_path,None)


def test_client_roundtrip_cleanup_and_failed_response_no_retry(tmp_path,monkeypatch,request):
    from multimodal_jev.span_rollout_rpc import RolloutClient
    from multimodal_jev import local_runtime_storage,local_checkpoint
    # Only storage admission is substituted (test tmpfs/overlay); actual Unix
    # connection, durable file writer, tensor load and message framing execute.
    monkeypatch.setattr(local_runtime_storage,'check_paths',lambda paths:[])
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda paths:[])
    # pytest tmp_path may exceed the AF_UNIX 107-byte pathname limit on hosts.
    import tempfile
    from pathlib import Path
    short=tempfile.TemporaryDirectory(prefix='jev-',dir='/tmp')
    request.addfinalizer(short.cleanup)
    address=Path(short.name)/'rpc.sock';listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    listener.bind(str(address));listener.listen(1)
    source,_,rows=payload(tmp_path);calls=[];errors=[]
    class Engine:
        def rollout(self,config,renderer):
            calls.append(config['adapter_id'])
            if len(calls)>1:raise ValueError('injected engine failure')
            return dict(passed=True,backend='vllm',base_id='base',adapter_id=config['adapter_id'])
    def worker():
        try:
            with listener:
                connection,_=listener.accept()
                with connection:serve_connection(connection,Engine(),tmp_path,None)
        except ValueError as exc:errors.append(str(exc))
    thread=threading.Thread(target=worker);thread.start()
    client=RolloutClient(address,tmp_path/'exchange',timeout=5)
    row=rows[0];renderer=[(dict(id=row['id']),row['inputs'],row['positions'],row['candidates'],
        row['request_id'],row['identity'],None)]
    config=dict(base_id='base',adapter_id='adapter')
    try:
        assert client.rollout(config,renderer)['passed']
        assert not list((tmp_path/'exchange').glob('*.pt'))
        with pytest.raises(ValueError,match='injected'):client.rollout(config,renderer)
        assert client.phase=='failed'
        with pytest.raises(ValueError):client.rollout(config,renderer)
        assert len(list((tmp_path/'exchange').glob('*.pt')))==1
    finally:client.close()
    thread.join(5)
    assert not thread.is_alive() and calls==['adapter','adapter'] and errors

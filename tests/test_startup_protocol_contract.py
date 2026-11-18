"""Real sockets: invalid peers must not become successful cleanup evidence."""
import socket
import struct
import threading

import pytest
from multimodal_jev.span_rollout_rpc import send, receive, MAX_MESSAGE_BYTES
from tools.reports.context_runtime_startup import close_replica


def peer(tmp_path, greeting, reply):
    address=tmp_path/'socket'
    server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    server.bind(str(address));server.listen(1)
    received=[]
    def run():
        try:
            with server.accept()[0] as connection:
                connection.settimeout(2)
                if greeting=='oversized':
                    connection.sendall(struct.pack('!I',MAX_MESSAGE_BYTES+1));return
                send(connection,greeting)
                try:
                    received.append(receive(connection));send(connection,reply)
                except (EOFError,ConnectionError,OSError):pass
        finally:server.close()
    thread=threading.Thread(target=run);thread.start()
    return address,thread,received


def hello(width=32):
    return dict(protocol='replica-macro-runtime-hello-v1',max_request_queries=width,
                runtime_identity={'padding':'x'*20000},policy_id='policy',contract={})


@pytest.mark.parametrize('width',[32,48,192])
def test_valid_greeting_capacity_is_protocol_data(tmp_path,width):
    ack=dict(sequence=0,ok=True,result=None)
    address,thread,received=peer(tmp_path,hello(width),ack)
    try:assert close_replica(address,2)==ack
    finally:thread.join(3)
    assert not thread.is_alive() and received==[dict(operation='close',sequence=0)]


@pytest.mark.parametrize('greeting,reply',[
    ([],{}),('oversized',{}),(hello(True),{}),
    (dict(hello(),protocol='unknown'),{}),
    (hello(),hello()),(hello(),dict(sequence=False,ok=1,result=None)),
])
def test_invalid_or_repeated_frames_rejected(tmp_path,greeting,reply):
    address,thread,_=peer(tmp_path,greeting,reply)
    try:
        with pytest.raises(ValueError):close_replica(address,2)
    finally:thread.join(3)
    assert not thread.is_alive()

"""Actual spawned worker/socket/file transport; substitute model and activation."""
import copy
import multiprocessing
import socket
import uuid
import pytest
from test_context_bulk_rpc_candidate import record
from test_context_bulk_session_candidate import full_item,setup,equal_output
from test_context_rollout_session import REF,P0,P1,BASE,CONTRACT,spec
from context_rollout_session import ContextRolloutSession
from tools.reports.context_bulk_session_candidate import BulkContextRolloutSession
from tools.reports.context_bulk_rpc_candidate import BulkContextEngineClient,serve_connection


def worker(address,exchange,ready):
    class Engine:
        bundle=dict(policy_id=P0,contract=CONTRACT)
        def score(self,inputs,positions,*,audio_token_id):
            result=record(self.bundle['policy_id'],dict(inputs=inputs,positions=positions))
            result['prob']={REF:[.7,.3],P0:[.4,.6],P1:[.8,.2]}[self.bundle['policy_id']]
            return result
        def switch(self,**kw):
            self.bundle=dict(self.bundle,policy_id=kw['policy_id'])
            return dict(policy_id=kw['policy_id'],head_marker_exact=True,native_loaded_tensors=112)
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as listener:
        listener.bind(address);listener.listen(1);ready.send('ready');ready.close()
        connection,_=listener.accept()
        with connection:connection.settimeout(30);serve_connection(connection,Engine(),exchange)


def test_spawned_service_session_two_updates(tmp_path,monkeypatch):
    import multimodal_jev.local_runtime_storage as storage
    import multimodal_jev.local_checkpoint as checkpoint
    import context_policy_activation as activation
    monkeypatch.setattr(storage,'check_paths',lambda p:None)
    monkeypatch.setattr(checkpoint,'check_paths',lambda p:None)
    monkeypatch.setattr(activation,'prepare_switch',lambda current,**kw:dict(current,policy_id=kw['policy_id']))
    original,_,_=setup(tmp_path/'original',monkeypatch,ContextRolloutSession)
    candidate,_,cache=setup(tmp_path/'candidate',monkeypatch,BulkContextRolloutSession)
    address='\0jev-bulk-test-'+uuid.uuid4().hex
    ctx=multiprocessing.get_context('spawn');parent,child=ctx.Pipe(duplex=False)
    process=ctx.Process(target=worker,args=(address,str(tmp_path/'exchange'),child));process.start();child.close()
    client=None
    try:
        assert parent.poll(30) and parent.recv()=='ready'
        client=BulkContextEngineClient(address,tmp_path/'exchange',bundle=dict(policy_id=P0,contract=CONTRACT),timeout=30)
        candidate.engine=client
        items=[full_item(P0,i) for i in range(127)];pad=copy.deepcopy(items[0])
        pad.update(request_id='padding',real_request=False,request_weight=0.);items.append(pad)
        a=original.prepare(items,update=0);b=candidate.prepare(items,update=0);equal_output(a,b)
        assert client.sequence==4 # switch reference, score reference, switch behavior, score behavior
        for s in (original,candidate):s.accept_actor_update(update=0,next_policy=spec(P1))
        for item in items:item['policy_id']=P1
        equal_output(original.prepare(items,update=1),candidate.prepare(items,update=1))
        assert client.sequence==6 and len(list(cache.namespace.glob('*.json')))==127
        assert not list((tmp_path/'exchange').glob('*.pt'))
    finally:
        if client is not None:client.close()
        parent.close();process.join(30)
        if process.is_alive():process.terminate();process.join(5);pytest.fail('Worker did not close')
    assert process.exitcode==0

import copy
from types import SimpleNamespace as NS
import pytest
import torch

from multimodal_jev.context_logits_wire import install_logits_readout,validate_logits_record,WIRE
from tools.reports.context_dual_rollout_candidate import encode_logits_many
from test_context_pooling import case as head_case
from test_context_batched_request_candidate import case as engine_case


def test_single_forward_actual_logits_unchanged_probabilities_and_activation_identity():
    r,h,lengths,prompts,extras=head_case();r.bfloat16();h=h.bfloat16()
    before=r(h,lengths,prompts,extras);state=copy.deepcopy(r.state_dict());identity=id(r)
    calls=[]
    handle=r.head.projection.register_forward_hook(lambda m,a,o:calls.append(o.detach().clone()))
    receipt=install_logits_readout(r);after=r(h,lengths,prompts,extras);handle.remove()
    assert id(r)==identity and len(calls)==2 and receipt['extra_head_forwards']==0
    for i,output in enumerate(after):
        assert torch.equal(output[:,:1],before[i])
        assert torch.equal(output[:,1:],calls[i][0].float())
    assert set(state)==set(r.state_dict())
    for key,v in state.items():assert torch.equal(v,r.state_dict()[key])
    # Existing activation holds this same object and mutates its policy ID.
    r.policy_id='b'*64
    with pytest.raises(ValueError):r(h,lengths,prompts,extras)
    assert not r._jev_logits_busy
    updated=[dict(e,policy_id='b'*64) for e in extras]
    assert len(r(h,lengths,prompts,updated))==2


def test_readout_no_cross_request_history_or_extra_projection_hooks():
    r,h,l,p,e=head_case();r.float();h=h.float();install_logits_readout(r)
    first=r(h,l,p,e);changed=h.clone();changed[:5]*=3
    second=r(changed,l,p,e)
    assert torch.equal(first[1],second[1])
    assert not r.head.projection._forward_hooks
    with pytest.raises(ValueError):install_logits_readout(r)


def dual_engine():
    owner,rows,calls=engine_case()
    def encode(prompts,**kw):
        calls.append(prompts)
        z=torch.tensor([[-1.],[1.]])
        return [NS(prompt_token_ids=p.extra_kwargs['expected_input_ids'],
                   outputs=NS(data=torch.cat((z.sigmoid(),z),dim=1))) for p in kw['pooling_params']]
    owner.engine=NS(encode=encode)
    return owner,rows,calls


def test_bulk_encode_retains_full_input_identity_with_logits():
    owner,rows,calls=dual_engine()
    engine=owner.engine
    got=encode_logits_many(owner,rows,audio_token_id=98,params_class=NS)
    assert len(calls)==1 and len(calls[0])==2 and owner.engine is engine
    for r in got:
        assert r['output_protocol']==WIRE and r['logits']==[-1.,1.]
        validate_logits_record(r)
    assert got[0]['request_id']!=got[1]['request_id']


@pytest.mark.parametrize('fault',['prob_only','wrong_logit','nan','wrong_dtype','reorder','policy'])
def test_wire_failures(fault):
    owner,rows,calls=dual_engine();original=owner.engine.encode
    def encode(*args,**kw):
        results=original(*args,**kw)
        if fault=='prob_only':results[0].outputs.data=results[0].outputs.data[:,:1]
        if fault=='wrong_logit':results[0].outputs.data[0,1]+=1
        if fault=='nan':results[0].outputs.data[0,1]=float('nan')
        if fault=='wrong_dtype':results[0].outputs.data=results[0].outputs.data.bfloat16()
        if fault=='reorder':results.reverse()
        if fault=='policy':owner.bundle=dict(owner.bundle,policy_id='b'*64)
        return results
    owner.engine.encode=encode
    with pytest.raises(ValueError):encode_logits_many(owner,rows,audio_token_id=98,params_class=NS)


def test_original_probability_contract_has_no_logits_without_opt_in():
    owner,rows,_=engine_case()
    with pytest.raises(ValueError):encode_logits_many(owner,rows,audio_token_id=98,params_class=NS)


def test_service_entrypoint_reuses_pinned_lifecycle():
    from tools.reports.serve_context_dual_candidate import service_namespace
    from tools.reports.context_dual_rollout_candidate import RUNTIME,owner_class
    ns=service_namespace()
    assert ns['PROTOCOL']=='bulk_dual_logits_candidate_v1'
    assert ns['PROFILE']==RUNTIME and ns['owner_class'] is owner_class


def _socket_process(address,exchange):
    import socket
    from tools.reports.context_batched_service_candidate import serve_connection
    owner,_,_=dual_engine()
    owner.bundle['contract']['base_id']='b'*64
    owner.score_many=lambda rows,**kw:encode_logits_many(owner,rows,params_class=NS,**kw)
    listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    listener.bind(address);listener.listen(1);listener.settimeout(10)
    try:
        connection,_=listener.accept()
        with connection:
            connection.settimeout(10);serve_connection(connection,owner,exchange)
    finally:listener.close()


def test_real_spawn_process_bulk_transport(tmp_path,monkeypatch):
    import multiprocessing
    import time
    import multimodal_jev.local_runtime_storage as storage
    import multimodal_jev.local_checkpoint as checkpoint
    from tools.reports.context_dual_rollout_candidate import DualContextEngineClient
    monkeypatch.setattr(storage,'check_paths',lambda _:None)
    monkeypatch.setattr(checkpoint,'check_paths',lambda _:None)
    owner,rows,_=dual_engine();owner.bundle['contract']['base_id']='b'*64
    address=str(tmp_path/'wire.sock')
    worker=multiprocessing.get_context('spawn').Process(target=_socket_process,args=(address,str(tmp_path)))
    worker.start();client=None
    try:
        startup_started=time.monotonic()
        deadline=startup_started+60
        while not (tmp_path/'wire.sock').exists():
            if not worker.is_alive() or time.monotonic()>deadline:raise RuntimeError('test worker not ready')
            time.sleep(.01)
        print('spawn_socket_ready_seconds',time.monotonic()-startup_started,flush=True)
        client=DualContextEngineClient(address,tmp_path,bundle=owner.bundle,timeout=10)
        # Deduplicated engine batch must restore all original rows and logits.
        values=client.score_many([rows[1],rows[0],copy.deepcopy(rows[1])],audio_token_id=98)
        assert len(values)==3 and values[0]==values[2]
        assert all(v['logits']==[-1.,1.] for v in values)
        assert client.sequence==1 and not list(tmp_path.glob('bulk-*.pt'))
        client.close();client=None;worker.join(10)
        assert worker.exitcode==0
    finally:
        if client:client.close()
        if worker.is_alive():worker.terminate();worker.join(5)

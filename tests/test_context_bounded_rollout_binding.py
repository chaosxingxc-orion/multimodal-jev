import copy
import pytest


def test_stage_binding_keeps_service32_and_sets_kernel8_explicitly():
    from test_prepare_context_stage2_runtime import inputs
    from tools.reports.prepare_context_stage2_runtime import configured
    a = inputs()
    execution = dict(kind='bounded_prefill_v1',max_requests=8,acoustic_batch_chunks=32)
    a[2]['attention_execution'] = execution
    _, services, _ = configured(*a)
    assert all(v['attention_execution']==execution and v['policy']['max_batch_queries']==32
               for v in services.values())
    a[2]['attention_execution']['max_requests']=1
    with pytest.raises(ValueError,match='attention8'):
        configured(*a)


def test_shared_service_keeps_scope_and_protocol_with_opt_in_owner(monkeypatch):
    from tools.reports import context_replica_runtime as runtime
    from tools.reports import serve_context_dual_candidate as service
    from tools.reports import context_bounded_rollout_candidate as bounded
    from multimodal_jev import cached_context_preparation as cache
    identity={'processor_sha256':'processor'};seen=[]
    monkeypatch.setattr(runtime,'verify_runtime',lambda c:identity)
    monkeypatch.setattr(cache,'processor_cache_identity',lambda p:'processor')
    sentinel=object()
    monkeypatch.setattr(service,'service_namespace',lambda:dict(run=sentinel,owner_class='old'))
    monkeypatch.setattr(bounded,'owner_class',lambda root,**kwargs:seen.append((root,kwargs)) or 'new')
    scope={'mounted_physical':[4,5,6,7]};module={'name':'validator','sha256':'hash'}
    c=dict(policy=dict(snapshot='policy',max_batch_queries=32,process_device_scope=scope,
                       device_scope_module=module),
           attention_execution=dict(kind='bounded_prefill_v1',max_requests=8,acoustic_batch_chunks=32))
    before=copy.deepcopy(c);ns=runtime.service_namespace(c)
    assert ns['run'] is sentinel and callable(ns['serve_connection'])
    assert ns['owner_class']('source')=='new'
    assert seen==[('source',dict(shared_scope=False,scope_module=module))]
    assert c==before
    c['attention_execution']['acoustic_batch_chunks']=1
    with pytest.raises(ValueError,match='attention8'):
        runtime.service_namespace(c)

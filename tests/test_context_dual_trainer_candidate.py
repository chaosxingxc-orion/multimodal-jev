import ast
import hashlib
import json
from pathlib import Path
import socket
import threading

import pytest

from tools.reports.context_dual_trainer_candidate import generate
from tools.reports import context_dual_runtime_candidate as runtime
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.span_rollout_rpc import send,receive


def test_trainer_preserves_optimizer_checkpoint_and_two_step_guard():
    original=Path('tools/jev/train_context_online_candidate.py').read_text()
    generated=generate(Path.cwd());ast.parse(generated)
    # Exact tail includes optimizer update, FP32 capture, BF16 export, all-rank
    # commit, activation and shutdown. Prefix edits only affect data/objective.
    marker="                actor=build_actor(cfg,wrapped,opt,collation=collation)"
    assert original.split(marker)[1]==generated.split(marker)[1]
    assert "if c['engineering_prefix'] is True and stop_update-start_update>2:" in generated
    assert 'BoundDualClient(' in generated and 'DualContextRolloutSession(' in generated
    assert 'items=bind_candidates(items,metadata,candidate_ids)' in generated
    assert 'dual_runtime_identity=runtime_identity' in generated
    assert "atomic_json(out/'checkpoint-scope.json',validate_trainable_scope(named))" in generated
    assert generated.index('validate_trainable_scope(named)') < generated.index(marker)
    assert "options=dict(stage='task'" not in generated


def make_runtime(tmp_path):
    root=tmp_path/'source';root.mkdir();(root/'code.py').write_text('x=1')
    files={'code.py':hashlib.sha256((root/'code.py').read_bytes()).hexdigest()}
    manifest=tmp_path/'RUNTIME.json';manifest.write_text(json.dumps(files))
    policy=tmp_path/'policy';processor=policy/'processor';processor.mkdir(parents=True)
    assets={}
    for name in ('tokenizer.json','tokenizer_config.json'):
        p=processor/name;p.write_text('{}');assets['processor/'+name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (policy/'MANIFEST.json').write_text(json.dumps(dict(files=assets)))
    identity=dict(image_sha256='52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c',
        source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),processor_sha256=content_hash(assets),
        profile=runtime.RUNTIME,output_protocol=runtime.WIRE)
    return dict(source_root=str(root),dual_runtime_manifest=str(manifest),source_policy=str(policy),dual_runtime_identity=identity)


@pytest.mark.parametrize('fault',[None,'source','extra','processor','manifest','profile'])
def test_runtime_requires_actual_complete_sources_and_processor(tmp_path,fault):
    c=make_runtime(tmp_path)
    if fault=='source':(Path(c['source_root'])/'code.py').write_text('x=2')
    if fault=='extra':(Path(c['source_root'])/'unexpected.py').write_text('')
    if fault=='processor':(Path(c['source_policy'])/'processor/tokenizer.json').write_text('{"changed":true}')
    if fault=='manifest':c['dual_runtime_identity']['source_manifest_sha256']='f'*64
    if fault=='profile':c['dual_runtime_identity']['profile']='other'
    if fault is None:assert runtime.verify_runtime(c)==c['dual_runtime_identity']
    else:
        with pytest.raises(ValueError):runtime.verify_runtime(c)


@pytest.mark.parametrize('fault',[None,'runtime','policy','protocol'])
def test_actual_socket_greeting_binds_connection_before_first_command(tmp_path,monkeypatch,fault):
    import multimodal_jev.local_runtime_storage as storage
    monkeypatch.setattr(storage,'check_paths',lambda _:None)
    c=make_runtime(tmp_path);identity=c['dual_runtime_identity'];bundle=dict(policy_id='a'*64,contract={'base_id':'b'*64})
    address=tmp_path/'s';listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    listener.bind(str(address));listener.listen(1);errors=[]
    def worker():
        try:
            conn,_=listener.accept()
            with conn:
                hello=dict(protocol='dual-runtime-hello-v1',runtime_identity=identity,**bundle)
                if fault=='runtime':hello['runtime_identity']=dict(identity,processor_sha256='f'*64)
                if fault=='policy':hello['policy_id']='c'*64
                if fault=='protocol':hello['protocol']='other'
                send(conn,hello)
                if fault is None:
                    message=receive(conn);assert message==dict(sequence=0,operation='close')
                    send(conn,dict(sequence=0,ok=True,result=None))
        except BaseException as exc:errors.append(exc)
        finally:listener.close()
    thread=threading.Thread(target=worker);thread.start()
    try:
        if fault is None:
            client=runtime.BoundDualClient(address,tmp_path,bundle=bundle,runtime_identity=identity)
            assert client.sequence==0 and client.runtime_identity==identity
            client.close()
        else:
            with pytest.raises(ValueError):runtime.BoundDualClient(address,tmp_path,bundle=bundle,runtime_identity=identity)
    finally:thread.join(5)
    assert not thread.is_alive() and not errors


def test_service_resume_accepts_same_processor_at_new_snapshot_and_emits_greeting(tmp_path,monkeypatch):
    import shutil
    from types import SimpleNamespace as NS
    from tools.reports import serve_context_dual_candidate as service
    c=make_runtime(tmp_path);resumed=tmp_path/'resume-policy'
    shutil.copytree(c['source_policy'],resumed);c['policy']=dict(snapshot=str(resumed))
    calls=[]
    monkeypatch.setattr(service,'service_namespace',lambda:dict(serve_connection=lambda *args:calls.append(args)))
    ns=runtime.service_namespace(c);a,b=socket.socketpair()
    engine=NS(bundle=dict(policy_id='current',contract={'base_id':'base'}))
    try:
        ns['serve_connection'](a,engine,'exchange')
        assert receive(b)==dict(protocol='dual-runtime-hello-v1',runtime_identity=c['dual_runtime_identity'],
                               policy_id='current',contract={'base_id':'base'})
        assert calls==[(a,engine,'exchange')]
    finally:a.close();b.close()
    (resumed/'processor/tokenizer.json').write_text('changed')
    with pytest.raises(ValueError):runtime.service_namespace(c)


def test_service_explicit_shared_scope_selects_runtime_validator(tmp_path,monkeypatch):
    from tools.reports import serve_context_dual_candidate as service
    from tools.reports import context_dual_rollout_candidate as rollout
    c=make_runtime(tmp_path)
    c['policy']=dict(snapshot=c['source_policy'],process_device_scope={
        'kind':'same_container_runtime_visibility_v1',
        'mounted_physical':[4,5,6,7],'visible_ordinal':'3'})
    calls=[]
    monkeypatch.setattr(service,'service_namespace',lambda:dict(serve_connection=lambda *args:None))
    monkeypatch.setattr(rollout,'owner_class',lambda root,**kw:calls.append((root,kw)))
    runtime.service_namespace(c)['owner_class']('payload')
    assert calls==[('payload',dict(shared_scope=True))]


def test_shared_owner_rejects_changed_device_validator(tmp_path,monkeypatch):
    from tools.reports import context_dual_rollout_candidate as rollout
    root=tmp_path/'tools/jev';root.mkdir(parents=True)
    (root/'context_shared_device_scope.py').write_text('def validate_tp1(c): return 7')
    monkeypatch.setattr(rollout,'transformed_source',lambda _: 'unused')
    with pytest.raises(ValueError,match='shared physical device validator changed'):
        rollout.owner_class(tmp_path,shared_scope=True)

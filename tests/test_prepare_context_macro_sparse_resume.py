"""Actual sparse package copying and sealing; model/container audits are fixtures."""
import json
from pathlib import Path

import pytest

from test_prepare_context_macro_replay import fixture as prefix_fixture
from test_context_macro_formal_candidate import config as formal_config
from tools.reports import prepare_context_macro_sparse_resume as m


def fixture(tmp_path,monkeypatch):
    c,parent,_=prefix_fixture(tmp_path)
    p=Path(parent['prepared'])/'package';base=Path(parent['operations_root']);run=base/'collected/run'
    t=json.loads((p/'training.json').read_text());t.update(formal_config())
    m.write(p/'training.json',t)
    for item in list(run.rglob('*')):
        if item.is_file():item.unlink()
    step=64;names=[f'rank-{r}/checkpoint-0064.pt' for r in range(2)]
    names+=['execution-0064.pt','policy-0064/MANIFEST.json','policy-0064/execution.pt',
            'adapter-0064/adapter.safetensors']
    for name in names:
        dest=run/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b'fixture')
    expected=m.expected_contract(t);expected['formal_experiment']=m.contract(t)
    commit=dict(updates=64,contract=expected,rank_shards={n:m.file_hash(run/n) for n in names[:2]},
        joint_policy_directory='policy-0064',joint_policy_manifest_sha256=m.file_hash(run/'policy-0064/MANIFEST.json'),
        joint_policy={'policy_id':'policy64'})
    m.write(run/'checkpoint-0064-committed.json',commit)
    (base/'results.tar').write_bytes(b'archive')
    m.write(base/'transport.json',dict(archive_sha256=m.file_hash(base/'results.tar'),
        files={str(f.relative_to(base/'collected')):m.file_hash(f) for f in run.rglob('*') if f.is_file()}))
    observations={}
    for role in ('preflight','client','service-0','service-1'):
        m.write(base/(role+'-launch.json'),dict(argv=[role],container_id=role))
        observations[role]=dict(Id=role,State=dict(Running=False,Status='exited',ExitCode=int(role=='client')))
    m.write(base/'collected-inspect.json',observations)
    (p/'FILES.json').unlink();m.write(p/'FILES.json',{str(f.relative_to(p)):m.file_hash(f) for f in p.rglob('*') if f.is_file()})
    parent['manifest_sha256']=m.verify(p)
    monkeypatch.setattr(m,'load_config',lambda *a:(parent,{}))
    monkeypatch.setattr(m,'validate_parent',lambda _:dict(files_sha256=m.verify(p)))
    monkeypatch.setattr(m,'command',lambda parent,role:[role])
    monkeypatch.setattr(m,'inspect_runtime',lambda *a:None)
    monkeypatch.setattr(m,'read_execution_bundle',lambda *a,**kw:{'policy_id':'policy64'})
    monkeypatch.setattr(m,'verify_native_adapter',lambda *a:None)
    monkeypatch.setattr(m,'verify_checkpoint',lambda *a,**kw:None)
    monkeypatch.setattr(m,'inspect_committed',lambda c:{'exact_bf16_from_master_shards':True})
    monkeypatch.setattr(m,'audit_prefix',lambda p,c,s:dict(
        status='formal_dp2_prefix_outputs_audited',start_update=0,stop_update=s,
        updates=[{} for _ in range(s)]))
    monkeypatch.setattr(m,'check_paths',lambda *a:None)
    # Exercise actual resume contract/pin/shard validation with fixture policy loader.
    from tools.reports import context_macro_formal_resume as runtime
    monkeypatch.setattr(runtime,'read_execution_bundle',lambda *a,**kw:{'policy_id':'policy64'})
    monkeypatch.setattr(runtime,'verify_native_adapter',lambda *a:None)
    c.update(parent_config='template',parent_local='local',step=step,
        transport_sha256=m.file_hash(base/'transport.json'))
    return c,parent


def test_failed_terminal_attempt_yields_delta_only_undispatched_resume(tmp_path,monkeypatch):
    c,parent=fixture(tmp_path,monkeypatch);source=Path(parent['prepared'])/'package'
    before={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    result=m.prepare(c,{})
    root=Path(c['output'])/'package';t=json.loads((root/'training.json').read_text())
    assert t['start_update']==64 and t['stop_update']==634
    assert t['model']=='/shared/model' and t['reference_cache']==c['remote_root']+'/reference-cache'
    assert (root/'RUNTIME-SOURCES.json').read_bytes()==before['RUNTIME-SOURCES.json']
    assert len(list((root/'resume').rglob('checkpoint-0064.pt')))==2
    for i in range(2):
        service=json.loads((root/f'service-{i}.json').read_text())
        assert service['policy']['policy_id']=='policy64'
        assert service['policy']['snapshot'].endswith('/resume/policy-0064')
    assert result['training_dispatch_allowed'] is False
    assert result['prefix_learning_history_verified'] is True
    assert result['files_sha256']==m.verify(root)
    assert before=={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}


@pytest.mark.parametrize('fault',['cursor','running','command','archive','transport','missing_shard',
    'changed_asset','objective','extra_policy','overlap','runtime_python','tensor_audit',
    'prefix_audit','unbound_history','prefix_cursor','preflight'])
def test_failed_or_unbound_recovery_cannot_be_sealed(tmp_path,monkeypatch,fault):
    c,parent=fixture(tmp_path,monkeypatch);base=Path(parent['operations_root']);run=base/'collected/run'
    if fault=='cursor':c['step']=317
    if fault=='running':
        p=base/'collected-inspect.json';v=json.loads(p.read_text());v['client']['State']['Running']=True;m.write(p,v)
    if fault=='preflight':
        p=base/'collected-inspect.json';v=json.loads(p.read_text());v['preflight']['State']['ExitCode']=1;m.write(p,v)
    if fault=='command':m.write(base/'client-launch.json',dict(argv=['other'],container_id='client'))
    if fault=='archive':(base/'results.tar').write_bytes(b'changed')
    if fault=='transport':c['transport_sha256']='0'*64
    if fault=='missing_shard':(run/'rank-1/checkpoint-0064.pt').unlink()
    if fault=='changed_asset':(run/'adapter-0064/adapter.safetensors').write_bytes(b'changed')
    if fault=='objective':
        p=run/'checkpoint-0064-committed.json';v=json.loads(p.read_text());v['contract']['global_batch']=192;m.write(p,v)
        p=base/'transport.json';v=json.loads(p.read_text());v['files']['run/checkpoint-0064-committed.json']=m.file_hash(run/'checkpoint-0064-committed.json');m.write(p,v);c['transport_sha256']=m.file_hash(p)
    if fault=='extra_policy':(run/'policy-0064/extra').write_bytes(b'not transported')
    if fault=='overlap':c['remote_root']=parent['remote_root']+'/nested'
    if fault=='runtime_python':
        p=Path(parent['prepared'])/'package/model_code.py';p.write_text('changed')
    if fault=='tensor_audit':
        def fail(*a,**kw):raise ValueError('tensor mismatch')
        monkeypatch.setattr(m,'verify_checkpoint',fail)
    if fault=='prefix_audit':
        def fail(*a,**kw):raise ValueError('invalid learning prefix')
        monkeypatch.setattr(m,'audit_prefix',fail)
    if fault=='unbound_history':(run/'rollout-0000.json').write_text('{}')
    if fault=='prefix_cursor':
        monkeypatch.setattr(m,'audit_prefix',lambda *a:dict(
            status='formal_dp2_prefix_outputs_audited',start_update=0,stop_update=128,updates=[{}]*128))
    with pytest.raises((ValueError,FileNotFoundError)):m.prepare(c,{})
    assert not (Path(c['output'])/'PREPARED.json').exists()

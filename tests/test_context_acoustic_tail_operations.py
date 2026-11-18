import json
from pathlib import Path
import pytest
from tools.reports import context_acoustic_tail_operations as a
from test_context_macro_training_operations import fixture as command_fixture
from test_prepare_context_macro_tail import fixture


@pytest.mark.parametrize('role',['client','preflight','service-0','service-1'])
def test_tail_keeps_repaired_cli_and_acoustic_services(tmp_path,role):
    c,_=command_fixture(tmp_path);old=a.tail.command(c,role);new=a.command(c,role)
    if role.startswith('service-'):
        assert len(old)==len(new)
        assert [(x,y) for x,y in zip(old,new) if x!=y]==[
            ('tools.reports.context_dual_runtime_candidate','tools.reports.context_acoustic_service_candidate')]
    else:assert old==new
    if role=='client':assert any(x.endswith('/context_macro_tail_entry_v2.py') for x in new)


def test_only_explicit_parent_variants():
    assert a.parent_validator('acoustic') is a.acoustic.validate
    assert a.parent_validator('kv16') is a.kv.validate
    with pytest.raises(ValueError):a.parent_validator('automatic')


def test_prepare_binds_exact_parent_and_never_dispatches(tmp_path,monkeypatch):
    c,source=fixture(tmp_path,monkeypatch);c['parent_kind']='kv16'
    receipt=dict(files_sha256='parent-hash',remote_root='/data/parent',ipc_root='/data/parent/ipc')
    monkeypatch.setattr(a.kv,'validate',lambda c:receipt)
    monkeypatch.setattr(a,'_base_validate',lambda c:None)
    seen=[];monkeypatch.setattr(a,'validate',lambda c:seen.append(c))
    before={p.name:p.read_bytes() for p in source.iterdir()}
    result=a.prepare(c,{})
    ops=json.loads((Path(c['output'])/'operations.json').read_text())
    assert ops['efficiency_parent']==json.loads(Path(c['parent_operations']).read_text())
    assert ops['parent_kind']=='kv16' and ops['project']==c['project']
    assert seen==[ops] and result['launched'] is False
    assert not ops['tail_diagnostic_dispatch_allowed'] and not ops['finite_training_dispatch_allowed']
    assert before=={p.name:p.read_bytes() for p in source.iterdir()}


@pytest.mark.parametrize('fault',[None,'service_capacity','actor','asset','file_set','parent'])
def test_full_delta_validator_rejects_undeclared_changes(tmp_path,monkeypatch,fault):
    c,source=fixture(tmp_path,monkeypatch);c['parent_kind']='kv16'
    path=Path(c['parent_operations']);parent=json.loads(path.read_text())
    parent.update(host='npu',lease='lease',model='/models/base',driver_root='/driver',shared_root='/shared',
                  manifest_sha256='parent-hash')
    path.write_text(json.dumps(parent))
    receipt=dict(files_sha256='parent-hash',remote_root='/data/parent',ipc_root='/data/parent/ipc')
    monkeypatch.setattr(a.kv,'validate',lambda c:receipt)
    # The base validator's manifest/runtime checks are tested separately; this
    # isolates the additional full parent/config difference checks.
    monkeypatch.setattr(a,'_base_validate',lambda c:dict(parent_manifest_sha256='parent-hash'))
    a.prepare(c,{})
    package=Path(c['output'])/'package';ops=json.loads((package.parent/'operations.json').read_text())
    if fault=='service_capacity':
        p=package/'service-0.json';v=json.loads(p.read_text());v['policy']={'kv_cache_memory_bytes':1};p.write_text(json.dumps(v))
    elif fault=='actor':
        p=package/'training.json';v=json.loads(p.read_text());v['microbatch']=16;p.write_text(json.dumps(v))
    elif fault=='asset':(package/'unchanged.py').write_text('modified')
    elif fault=='file_set':(package/'extra.py').write_text('new')
    elif fault=='parent':ops['efficiency_parent']['manifest_sha256']='other'
    if fault:
        with pytest.raises(ValueError):a.validate(ops)
    else:assert a.validate(ops)['parent_manifest_sha256']=='parent-hash'

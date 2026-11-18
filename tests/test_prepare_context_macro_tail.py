import json
from pathlib import Path

import pytest

from tools.reports import prepare_context_macro_tail as m


def fixture(tmp_path,monkeypatch):
    parent=tmp_path/'parent';source=parent/'package';source.mkdir(parents=True)
    old='/data/parent';runtime={'source_manifest_sha256':'old','profile':'fixture'}
    t=dict(start_update=0,stop_update=100,microbatch=8,experiment_kind='efficiency',
        global_batch=128,stage='task',source_root=old+'/package',
        pins={old+'/package/unchanged.py':'old','/shared/data':'data-hash'},
        dual_runtime_identity=runtime,macro_client={'runtime_identity':runtime})
    for name,obj in [('training.json',t),('FILES.json',{}),('RUNTIME-SOURCES.json',{}),
                     *[(f'service-{i}.json',{'pins':{},'dual_runtime_identity':runtime}) for i in range(2)]]:
        (source/name).write_text(json.dumps(obj))
    (source/'unchanged.py').write_text('sentinel=1\n')
    ops=dict(prepared=str(parent),remote_root=old)
    op=tmp_path/'parent.json';op.write_text(json.dumps(ops))
    receipt=dict(files_sha256='parent-hash',remote_root=old,ipc_root=old+'/ipc')
    monkeypatch.setattr(m,'validate_parent',lambda c:receipt)
    monkeypatch.setattr(m,'check_paths',lambda c:None)
    monkeypatch.setattr(m,'validate',lambda c:None)
    c=dict(parent_operations=str(op),output=str(tmp_path/'output'),remote_root='/data/tail',
        project=str(Path.cwd()),operations_root=str(tmp_path/'ops'),tag='tail')
    return c,source


def test_prepare_preserves_parent_bytes_seals_new_identity_and_disables_launch(tmp_path,monkeypatch):
    c,source=fixture(tmp_path,monkeypatch)
    before={p.name:p.read_bytes() for p in source.iterdir()}
    result=m.prepare(c,{})
    assert result['launched'] is False and result['tail_admission']['real_queries']==[65,128]
    assert before=={p.name:p.read_bytes() for p in source.iterdir()}
    package=Path(c['output'])/'package';t=json.loads((package/'training.json').read_text())
    assert (package/'unchanged.py').read_bytes()==before['unchanged.py']
    assert t['pins']['/shared/data']=='data-hash'
    assert t['stop_update']==2 and t['experiment_kind']=='tail_admission'
    assert t['dual_runtime_identity']['source_manifest_sha256']==m.file_hash(package/'RUNTIME-SOURCES.json')
    assert m.verify(package)==result['manifest_sha256']
    ops=json.loads((package.parent/'operations.json').read_text())
    assert ops['finite_training_dispatch_allowed'] is False
    assert ops['tail_diagnostic_dispatch_allowed'] is False
    with pytest.raises(ValueError,match='new disjoint'):m.prepare(c,{})


@pytest.mark.parametrize('remote',['/data/parent/child','/data','relative','/data/../tail'])
def test_overlapping_or_invalid_remote_rejected_before_copy(tmp_path,monkeypatch,remote):
    c,_=fixture(tmp_path,monkeypatch);c['remote_root']=remote
    with pytest.raises(ValueError,match='new disjoint'):m.prepare(c,{})
    assert not Path(c['output']).exists()

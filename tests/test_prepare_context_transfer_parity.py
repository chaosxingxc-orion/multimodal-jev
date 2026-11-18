"""Real package sealing; policy tensors and full input admission are fixtures."""
import json
from pathlib import Path

import pytest

from test_prepare_context_macro_replay import fixture
from tools.reports import prepare_context_transfer_parity as m
from tools.reports import context_transfer_parity_operations as ops
from multimodal_jev import context_policy_artifacts


def setup(tmp_path,monkeypatch):
    _,parent,_=fixture(tmp_path);parent['lease']='/lease'
    source=Path(parent['prepared'])/'package'
    (source/'policy').mkdir();(source/'adapter').mkdir()
    t=json.loads((source/'training.json').read_text())
    m.write(source/'policy/MANIFEST.json',dict(policy_id=t['source_policy_id']))
    (source/'adapter/weights').write_bytes(b'only delta')
    for i in range(2):
        p=source/f'service-{i}.json';s=json.loads(p.read_text())
        s.update(source_root=parent['remote_root']+'/package',source_policy=t['source_policy'])
        s['policy'].update(model=parent['model'],base_id=t['base_id'],max_decisions=60,
            max_batch_queries=32,max_batch_tokens=131072)
        m.write(p,s)
    receipt=json.loads((source.parent/'PREPARED.json').read_text())
    receipt['runtime_identity']['processor_sha256']='processor'
    monkeypatch.setattr(m,'validate_parent',lambda _:receipt)
    monkeypatch.setattr(m,'check_paths',lambda _:None)
    admitted=dict(dataset='MINDS_native14',candidate_ids=list(range(14)),population=[{}]*259,processor_sha256='processor')
    monkeypatch.setattr(m,'load_inputs',lambda _:admitted)
    monkeypatch.setattr(ops,'load_inputs',lambda _:admitted)
    monkeypatch.setattr(context_policy_artifacts,'read_execution_bundle',lambda *a,**kw:kw)
    monkeypatch.setattr(context_policy_artifacts,'verify_native_adapter',lambda *a:None)
    inputs=tmp_path/'inputs';inputs.mkdir();m.write(inputs/'MANIFEST.json',{})
    parent_path=tmp_path/'parent.json';m.write(parent_path,parent)
    c=dict(parent_operations=str(parent_path),project=str(Path.cwd()),output=str(tmp_path/'prepared'),
        operations_root=str(tmp_path/'operations'),remote_root='/data/parity',tag='parity',
        portable_parity_inputs=dict(root=str(inputs),manifest_sha256=m.file_hash(inputs/'MANIFEST.json'),panel_sha256='panel',cache_root='/data/cache'))
    return c,source


def test_seal_keeps_runtime_and_delta_weights_binds_both_services(tmp_path,monkeypatch):
    c,source=setup(tmp_path,monkeypatch);before={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    result=m.prepare(c,{})
    root=Path(c['output']);package=root/'package';o=json.loads((root/'operations.json').read_text())
    assert result['launched'] is False and o['finite_inference_dispatch_allowed'] is False
    assert ops.validate(o)['recordings']==259
    assert before=={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert (package/'model_code.py').read_bytes()==before['model_code.py']
    assert (package/'adapter/weights').read_bytes()==b'only delta'
    assert not (package/'training.json').exists()
    for i in range(2):
        s=json.loads((package/f'service-{i}.json').read_text())
        assert s['policy']['max_decisions']==14 and s['policy']['max_batch_queries']==32
    with pytest.raises(ValueError,match='new disjoint'):m.prepare(c,{})


@pytest.mark.parametrize('fault',['width','device','cache','policy','pins'])
def test_resealed_mismatch_rejected(tmp_path,monkeypatch,fault):
    c,_=setup(tmp_path,monkeypatch);m.prepare(c,{})
    root=Path(c['output']);p=root/'package';o=json.loads((root/'operations.json').read_text())
    path=p/'service-1.json';s=json.loads(path.read_text())
    if fault=='width':s['policy']['max_decisions']=60
    if fault=='device':s['policy']['physical_devices']=[4]
    if fault=='policy':s['policy']['policy_id']='other'
    if fault=='pins':s['pins']={}
    if fault=='cache':o['cache_root']='/wrong/cache'
    m.write(path,s)
    m.write(p/'FILES.json',{str(f.relative_to(p)):m.file_hash(f) for f in p.rglob('*') if f.is_file() and f.name!='FILES.json'})
    receipt=json.loads((root/'PREPARED.json').read_text());receipt['files_sha256']=m.verify(p)
    o['manifest_sha256']=receipt['files_sha256'];m.write(root/'PREPARED.json',receipt)
    with pytest.raises(ValueError):ops.validate(o)

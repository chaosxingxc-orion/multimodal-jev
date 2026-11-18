import json
import shutil

import pytest

from test_context_transfer_macro_inputs import fixture
from tools.reports import context_transfer_parity_inputs as m
from tools.reports.context_transfer_macro_inputs import normalize
from tools.reports.context_transfer_macro_parity import panel


def build(tmp_path,monkeypatch):
    from multimodal_jev import local_checkpoint
    rows,receipts,requests,ids=fixture(14)
    for r in receipts:
        r.update(cache_key='key'+r['id'],cache_sha256='cache')
        r['tensors']['input_ids']['shape']=[1,100]
        r['tensors']['input_features']['shape']=[1,128,200]
    population,independent=normalize(rows,receipts,requests,ids,expected_count=3,decisions=14)
    admitted=dict(dataset='MINDS_native14',population=population,input_receipts=independent,candidate_ids=ids,
        source_rows=rows,cache_receipts=receipts,requests={r['id']:q for r,q in zip(rows,requests,strict=True)},
        processor_sha256='processor',cache_root='/remote/cache')
    expected=tmp_path/'expected.json'
    expected.write_text(json.dumps(panel(admitted),ensure_ascii=False,indent=2)+'\n')
    monkeypatch.setattr(m,'load_full',lambda _:admitted)
    monkeypatch.setattr(m,'check_paths',lambda _:None)
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    c=dict(evaluation={},panel=dict(path=str(expected),sha256=m.file_hash(expected)),output=str(tmp_path/'bundle'))
    r=m.prepare(c,{})
    return c,dict(root=c['output'],manifest_sha256=r['manifest_sha256'],
        panel_sha256=c['panel']['sha256'],cache_root='/new/cache'),admitted


def test_move_bundle_delete_source_and_bind_new_cache_root(tmp_path,monkeypatch):
    c,spec,admitted=build(tmp_path,monkeypatch)
    moved=tmp_path/'moved';shutil.move(c['output'],moved);(tmp_path/'expected.json').unlink()
    spec['root']=str(moved);loaded=m.load(spec)
    assert loaded['population']==admitted['population']
    assert loaded['input_receipts']==admitted['input_receipts']
    assert loaded['requests']==admitted['requests']
    assert loaded['cache_root']=='/new/cache'


@pytest.mark.parametrize('fault',['request','position','cache','gold','panel','symlink'])
def test_resealed_corruption_still_checked_against_independent_panel(tmp_path,monkeypatch,fault):
    c,spec,_=build(tmp_path,monkeypatch);root=tmp_path/'bundle'
    name='requests.json' if fault=='request' else 'rows.json' if fault=='gold' else 'cache-receipts.json'
    if fault=='panel':name='panel.json'
    p=root/name;data=json.loads(p.read_text())
    if fault=='request':data[0]['request']['audio']='/other.wav'
    elif fault=='position':data[0]['positions'][0][0]=4
    elif fault=='cache':data[0]['audio_sha256']='other'
    elif fault=='gold':data[0]['gold']='hidden'
    elif fault=='panel':data['assets'][0]['audio_sha256']='other'
    if fault=='symlink':
        target=tmp_path/'outside.json';p.rename(target);p.symlink_to(target)
    else:p.write_text(json.dumps(data))
    manifest=json.loads((root/'MANIFEST.json').read_text());manifest['files'][name]=m.file_hash(p)
    if fault=='panel':manifest['panel_sha256']=m.file_hash(p);manifest['panel_content_sha256']=m.digest(data)
    (root/'MANIFEST.json').write_text(json.dumps(manifest));spec['manifest_sha256']=m.file_hash(root/'MANIFEST.json')
    with pytest.raises(ValueError):m.load(spec)

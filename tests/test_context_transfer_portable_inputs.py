import json
import shutil
from pathlib import Path

import pytest

from test_context_transfer_parity_inputs import build as panel_fixture
from tools.reports import context_transfer_portable_inputs as m


def build(tmp_path,monkeypatch):
    _,_,admitted=panel_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'SIZES',dict(m.SIZES,MINDS_native14=(3,14)))
    monkeypatch.setattr(m,'load_original',lambda c:admitted)
    monkeypatch.setattr(m,'check_paths',lambda c:None)
    root=tmp_path/'evidence';root.mkdir()
    for name,value in [('population.json',admitted['population']),('candidate-ids.json',admitted['candidate_ids']),
                       ('input-receipts.json',admitted['input_receipts'])]:
        (root/name).write_text(json.dumps(value))
    (root/'MANIFEST.json').write_text(json.dumps(dict(dataset='MINDS_native14',recordings=3,decisions=14,
        processor_sha256='processor',artifacts={p.name:m.file_hash(p) for p in root.iterdir()})))
    c=dict(evaluation={},population_evidence=dict(root=str(root),manifest_sha256=m.file_hash(root/'MANIFEST.json')),
        output=str(tmp_path/'full'))
    result=m.prepare(c,{})
    return c,result,admitted


def test_full_inputs_relocate_without_original_dataset_or_development_files(tmp_path,monkeypatch):
    c,spec,admitted=build(tmp_path,monkeypatch)
    shutil.rmtree(c['population_evidence']['root'])
    target=tmp_path/'moved';shutil.move(spec['root'],target)
    spec.update(root=str(target),cache_root='/other/cache')
    loaded=m.load(spec)
    for k in ('population','input_receipts','requests','candidate_ids'):assert loaded[k]==admitted[k]
    assert loaded['cache_root']=='/other/cache'


@pytest.mark.parametrize('fault',['request','receipt','population','gold','receipt_gold','subset','extra','symlink','source_release'])
def test_resealed_corruption_cannot_replace_independent_population(tmp_path,monkeypatch,fault):
    c,spec,_=build(tmp_path,monkeypatch);root=Path(spec['root'])
    name='requests.json' if fault=='request' else 'cache-receipts.json' if fault in ('receipt','receipt_gold') else 'rows.json'
    if fault=='population':name='population-evidence/population.json'
    value=json.loads((root/name).read_text())
    if fault=='request':value[0]['request']['audio']='/changed.wav'
    if fault=='receipt':value[0]['positions'][0][0]+=1
    if fault=='population':value[0]['audio_sha256']='changed'
    if fault=='gold':value[0]['gold']='forbidden'
    if fault=='receipt_gold':value[0]['gold']='forbidden'
    if fault=='subset':value.pop()
    (root/name).write_text(json.dumps(value))
    if fault=='extra':(root/'gold.json').write_text('{}')
    if fault=='symlink':
        p=root/'rows.json';p.rename(tmp_path/'outside');p.symlink_to(tmp_path/'outside')
    manifest=json.loads((root/'MANIFEST.json').read_text())
    manifest['files'][name]=m.file_hash(root/name)
    if fault=='population':
        p=root/'population-evidence/MANIFEST.json';v=json.loads(p.read_text())
        v['artifacts']['population.json']=m.file_hash(root/name);p.write_text(json.dumps(v))
        manifest['files']['population-evidence/MANIFEST.json']=m.file_hash(p)
        manifest['population_evidence_sha256']=m.file_hash(p)
    if fault=='source_release':manifest['source_release']={'sha256':'other'}
    (root/'MANIFEST.json').write_text(json.dumps(manifest));spec['manifest_sha256']=m.file_hash(root/'MANIFEST.json')
    with pytest.raises(ValueError):m.load(spec)


def test_original_admission_must_match_independent_population(tmp_path,monkeypatch):
    c,spec,admitted=build(tmp_path,monkeypatch)
    admitted['population'][0]['audio_sha256']='changed';c['output']=str(tmp_path/'bad')
    with pytest.raises(ValueError):m.prepare(c,{})
    assert not Path(c['output']).exists()


@pytest.mark.parametrize('has_release',[False,True])
def test_clean_test_requires_explicit_current_source_release(tmp_path,monkeypatch,has_release):
    c,_,admitted=build(tmp_path,monkeypatch)
    monkeypatch.setitem(m.SIZES,'SLURP_clean_TEST60',(3,14))
    admitted['dataset']='SLURP_clean_TEST60'
    source=dict(path='/source/release',sha256='source-hash')
    c['evaluation']=dict(dataset='SLURP_clean_TEST60',source_release=source)
    root=Path(c['population_evidence']['root']);p=root/'MANIFEST.json'
    manifest=json.loads(p.read_text());manifest['dataset']='SLURP_clean_TEST60'
    if has_release:manifest['source_release']=source
    p.write_text(json.dumps(manifest));c['population_evidence']['manifest_sha256']=m.file_hash(p)
    c['output']=str(tmp_path/'slurp')
    if not has_release:
        with pytest.raises(ValueError,match='source held-out ontology'):m.prepare(c,{})
        assert not Path(c['output']).exists()
    else:
        spec=m.prepare(c,{})
        assert m.load(spec)['source_release']==source

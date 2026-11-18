import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from test_slurp_official_training import fixture
from multimodal_jev.baseline import file_hash
from multimodal_jev import slurp_official_training as legacy
from multimodal_jev import slurp_portable_release as m


def test_migration_survives_removal_of_original_sources_and_preserves_every_row(tmp_path):
    origin=tmp_path/'origin';origin.mkdir();c,v1=fixture(origin);legacy.publish(c,v1)
    original={role:legacy.load(v1,file_hash(v1/'MANIFEST.json'),role,purpose=purpose)[:2]
        for role,purpose in (('learning','optimize'),('selection','evaluate'))}
    out=tmp_path/'v2';manifest=m.migrate(v1,file_hash(v1/'MANIFEST.json'),out)
    for name,h in manifest['migration']['unchanged_payload_sha256'].items():assert file_hash(out/name)==h
    moved=tmp_path/'relocated';out.rename(moved);shutil.rmtree(origin)
    for role,purpose in (('learning','optimize'),('selection','evaluate')):
        assert m.load(moved,file_hash(moved/'MANIFEST.json'),role,purpose=purpose)[:2]==original[role]
    spec=importlib.util.spec_from_file_location('independent',Path(__file__).resolve().parents[1]/'tools/jev/verify_slurp_official_intent.py')
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    assert verifier.verify(moved,file_hash(moved/'MANIFEST.json'))['passed']
    evidence=json.loads((moved/'publication-config.json').read_text())
    assert all(not Path(p).is_absolute() for p in evidence['pins'])
    assert all(not Path(p).is_absolute() for pair in evidence['reviews'].values() for p in pair)


def test_default_publication_is_portable_and_legacy_loading_is_preserved(tmp_path):
    c,v1=fixture(tmp_path);legacy.publish(c,v1)
    assert m.load(v1,file_hash(v1/'MANIFEST.json'),'learning',purpose='optimize')[0]
    out=tmp_path/'new';assert m.publish(c,out)['schema']==m.SCHEMA
    with pytest.raises(FileExistsError):m.publish(c,out)
    with pytest.raises(ValueError,match='selection cannot optimize'):m.load(out,file_hash(out/'MANIFEST.json'),'selection',purpose='optimize')


@pytest.mark.parametrize('fault',['traversal','absolute','symlink','evidence','label','population'])
def test_changed_evidence_and_rehashed_wrong_labels_refused(tmp_path,fault):
    c,_=fixture(tmp_path);out=tmp_path/'v2';m.publish(c,out)
    if fault in ('traversal','absolute'):
        with pytest.raises(ValueError):m.contained(out,'../tasks.json' if fault=='traversal' else '/tmp/tasks.json')
        return
    if fault=='symlink':(out/'unexpected').symlink_to(tmp_path/'tasks.json')
    if fault=='evidence':
        evidence=json.loads((out/'publication-config.json').read_text())
        (out/evidence['reviews']['learning'][1]).write_text('{}')
    if fault in ('label','population'):
        p=out/'learning.jsonl';rows=[json.loads(line) for line in p.read_text().splitlines()]
        if fault=='label':rows[0]['intent']='calendar_set'
        else:rows.pop()
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        manifest=json.loads((out/'MANIFEST.json').read_text());manifest['files']['learning.jsonl']=file_hash(p)
        (out/'MANIFEST.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):m.load(out,file_hash(out/'MANIFEST.json'),'learning',purpose='optimize')


def test_audio_hours_allow_only_cross_python_summation_roundoff(tmp_path):
    c,_=fixture(tmp_path);out=tmp_path/'v2';m.publish(c,out)
    path=out/'MANIFEST.json';manifest=json.loads(path.read_text())
    manifest['counts']['learning']['audio_hours']+=1e-12
    path.write_text(json.dumps(manifest))
    assert m.load(out,file_hash(path),'learning',purpose='optimize')[0]
    manifest['counts']['learning']['audio_hours']+=1e-4
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='reported counts'):m.load(out,file_hash(path),'learning',purpose='optimize')


def test_published_bundle_readable_by_collaborating_users(tmp_path):
    import stat
    c,_=fixture(tmp_path);out=tmp_path/'v2';m.publish(c,out)
    for path in [out,*out.rglob('*')]:
        mode=stat.S_IMODE(path.stat().st_mode)
        assert mode==(0o755 if path.is_dir() else 0o644)

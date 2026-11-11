import json
from pathlib import Path

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.slurp_training_release import publish
from test_slurp_training import fixture


def release_fixture(tmp_path):
    request,target,tasks,reviews=fixture()
    source=tmp_path/'source';source.mkdir()
    audio=tmp_path/'source.wav';audio.write_bytes(b'fixture-only-audio')
    request['input']['audio']=str(audio);request['audio_sha256']=file_hash(audio)
    target['request_sha256']=content_hash(request)
    for review in reviews.values():
        review.update(request_sha256=content_hash(request),native_target_sha256=content_hash(target))
    (source/'learning.requests.jsonl').write_bytes(canonical(request))
    (source/'learning.targets.jsonl').write_bytes(canonical(target))
    (source/'MANIFEST.json').write_bytes(canonical(dict(schema='slurp-clean-source-bundles-v1',
        files={p.name:file_hash(p) for p in source.iterdir()})))
    receipt=tmp_path/'receipt.json';receipt.write_bytes(canonical(dict(
        release=str(source),manifest_sha256=file_hash(source/'MANIFEST.json'))))
    tasks_path=tmp_path/'tasks.json';tasks_path.write_bytes(canonical(tasks))
    evidence=tmp_path/'evidence.json';evidence.write_text('fixture semantic evidence')
    ledger_path=tmp_path/'ledger.json';ledger_path.write_bytes(canonical(dict(
        source_manifest_sha256=file_hash(source/'MANIFEST.json'),tasks_sha256=file_hash(tasks_path),
        evidence_files={str(evidence):file_hash(evidence)},scope='synthetic fixture only',
        records=[dict(id=request['id'],reviews=reviews)])))
    return source,receipt,tasks_path,ledger_path,tmp_path/'out'


def test_publish_immutable_partial_review_release(tmp_path):
    args=release_fixture(tmp_path)
    ledger=json.loads(args[3].read_text());ledger['records'][0]['reviews']['date']={'status':'conflict'}
    args[3].write_bytes(canonical(ledger))
    path,report=publish(*args)
    assert report['decisions']==2 and report['counts']['conflict']==1
    assert report['reviewed_recordings']==1 and report['source_groups']==1
    rows=[json.loads(line) for line in (path/'train.jsonl').read_text().splitlines()]
    assert [row['target'] for row in rows]==['a',True]
    assert (path/'train.jsonl').stat().st_mode & 0o777==0o600
    for name,digest in report['files'].items():assert file_hash(path/name)==digest
    with pytest.raises(FileExistsError):publish(*args)


@pytest.mark.parametrize('mutation',['source','evidence','tasks','audio','outside','duplicate','sparse','empty'])
def test_invalid_release_refused_and_no_partial_output(tmp_path,mutation):
    args=release_fixture(tmp_path);source,_,tasks,ledger_path,out=args
    ledger=json.loads(ledger_path.read_text())
    if mutation=='source':(source/'learning.targets.jsonl').write_text('changed')
    elif mutation=='evidence':Path(next(iter(ledger['evidence_files']))).write_text('changed')
    elif mutation=='tasks':tasks.write_text('{}')
    elif mutation=='audio':(tmp_path/'source.wav').write_bytes(b'changed')
    elif mutation=='outside':ledger['records'][0]['id']='selection:other'
    elif mutation=='duplicate':ledger['records']*=2
    elif mutation=='sparse':del ledger['records'][0]['reviews']['date']
    else:
        for review in ledger['records'][0]['reviews'].values():review['status']='pending'
    ledger_path.write_bytes(canonical(ledger))
    with pytest.raises(ValueError):publish(*args)
    assert not out.exists() or list(out.iterdir())==[]

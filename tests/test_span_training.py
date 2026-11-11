import json
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.span_training import load_reviewed_bundles, target_tokens
from multimodal_jev.slurp_training import project, content_hash
from test_slurp_training import fixture


def make_release(tmp_path):
    request, target, tasks, reviews = fixture()
    audio = tmp_path / 'sample.wav'
    audio.write_bytes(b'fixture audio hash only')
    request['input']['audio'] = str(audio)
    request['audio_sha256'] = file_hash(audio)
    target['request_sha256'] = content_hash(request)
    for r in reviews.values():
        r['request_sha256'] = content_hash(request)
        r['native_target_sha256'] = content_hash(target)
    reviews['date']['status'] = 'conflict'
    examples, _ = project(request,target,tasks,reviews,'b'*64)
    release = tmp_path/'release'
    release.mkdir()
    (release/'tasks.json').write_text(json.dumps(tasks))
    (release/'train.jsonl').write_text(''.join(json.dumps(e.to_record())+'\n' for e in examples))
    manifest = dict(schema='slurp-reviewed-training-release-v1', reviewed_recordings=1, decisions=2,
                    files={p.name:file_hash(p) for p in release.iterdir()})
    (release/'MANIFEST.json').write_text(json.dumps(manifest))
    return release


def test_reviewed_bundle_keeps_conflict_mask_and_targets_outside_input(tmp_path):
    r = make_release(tmp_path)
    bundles,_ = load_reviewed_bundles(r,file_hash(r/'MANIFEST.json'),['intent','person','date'])
    b = bundles[0]
    assert b['targets'] == ['a',True,None]
    assert b['supervised'] == [True,True,False]
    assert 'targets' not in b['request'].model_input()
    assert 'slurp_review' not in json.dumps(b['request'].model_input())
    maps = [dict(values=['a','b'],token_ids=[10,11]),dict(values=[False,True],token_ids=[10,11]),
            dict(values=[False,True],token_ids=[10,11])]
    assert target_tokens(b,maps) == [10,11,-100]
    b['targets'][1] = 1
    with pytest.raises(ValueError,match='absent'):target_tokens(b,maps)


def test_release_mutation_rejected(tmp_path):
    r=make_release(tmp_path)
    pin=file_hash(r/'MANIFEST.json')
    with (r/'train.jsonl').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='file changed'):
        load_reviewed_bundles(r,pin,['intent','person','date'])

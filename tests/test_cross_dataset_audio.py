import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from audit_cross_dataset_audio import audit,compare
from multimodal_jev.baseline import file_hash


def test_overlap_counts_requests_not_audio_occurrences():
    cells={'a/train':{'a'*64:{'r1','r2'},'b'*64:{'r2'}},
           'b/test':{'a'*64:{'x1'},'b'*64:{'x1'},'c'*64:{'x2'}}}
    assert compare(cells)==[dict(left='a/train',right='b/test',shared_audio_hashes=2,
        left_affected_requests=2,right_affected_requests=1)]


@pytest.mark.parametrize('change',[None,'manifest','requests','population','hash','labels','split','duplicate'])
def test_complete_input_only_manifest_binding(tmp_path,change):
    root=tmp_path/'release';root.mkdir()
    row=dict(id='a',input=dict(audio='not-opened.wav'),audio_sha256='a'*64,source_group='g',split='train')
    if change=='hash':row['audio_sha256']='bad'
    if change=='labels':row['target']='forbidden'
    if change=='split':row['split']='test'
    path=root/'train.requests.jsonl';path.write_text(json.dumps(row)+'\n')
    if change=='duplicate':path.write_text(path.read_text()*2)
    manifest=dict(identity=dict(dataset='native'),release_id='release',splits=dict(train=dict(
        requests=2 if change=='population' else 1,source_groups=1,requests_sha256=file_hash(path))))
    mp=root/'MANIFEST.json';mp.write_text(json.dumps(manifest))
    plan=dict(schema='native-audio-overlap-plan-v1',datasets=[dict(dataset='native',release=str(root),manifest_sha256=file_hash(mp))])
    if change=='manifest':mp.write_text('{}')
    if change=='requests':path.write_text('{}')
    if change is None:
        result=audit(plan);assert result['counts']['native/train']['unique_audio_hashes']==1
        assert result['targets_read'] is False and result['pairs']==[]
    else:
        with pytest.raises(ValueError):audit(plan)

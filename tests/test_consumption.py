import json
from dataclasses import replace
import pytest
from multimodal_jev.consumption import request_from_example, validate_request, inference_messages, export_release
from multimodal_jev.task_schema import Candidate, DecisionSchema, JevExample


def example(split='train'):
    return JevExample(split+'-1', split, '/audio.wav',
        DecisionSchema('task','choice','Choose the command', (Candidate('a','on'),Candidate('b','off'))),
        'a', {'source_group': split+'-g','audio_sha256':split+'-hash','variant':'native','transcript_reference':'secret'})


def test_label_and_reference_cannot_change_actor_request():
    ex=example(); other=replace(ex,target='b',metadata={**ex.metadata,'transcript_reference':'different'})
    assert request_from_example(ex)==request_from_example(other)
    request=request_from_example(ex)
    assert 'secret' not in json.dumps(inference_messages(request,'turn on','decide'))
    with pytest.raises(ValueError): validate_request({**request,'target':'a'})
    with pytest.raises(ValueError): validate_request({**request,'input':{**request['input'],'transcript_reference':'secret'}})


def test_export_full_coverage_and_separate_targets(tmp_path):
    manifests={}
    for split in ('train','valid','test'):
        path=tmp_path/(split+'.jsonl');path.write_text(json.dumps(example(split).to_record())+'\n');manifests[split]=path
    out,receipt=export_release(manifests,tmp_path/'output','fixture-jev')
    assert {s:r['requests'] for s,r in receipt['splits'].items()}==dict(train=1,valid=1,test=1)
    assert 'transcript_reference' not in (out/'train.requests.jsonl').read_text()
    assert 'transcript_reference' in (out/'train.targets.jsonl').read_text()
    with pytest.raises(FileExistsError):export_release(manifests,tmp_path/'output','fixture-jev')
    train=example('test');train=replace(train,metadata={**train.metadata,'audio_sha256':'train-hash'})
    manifests['test'].write_text(json.dumps(train.to_record())+'\n')
    with pytest.raises(ValueError,match='audio crosses'):export_release(manifests,tmp_path/'output','fixture-jev')


def test_scoring_invalid_null_and_complete_coverage():
    from multimodal_jev.consumption_score import score
    ex=example();request=request_from_example(ex)
    targets=[{'id':ex.example_id,'target':None,'metadata':{}}]
    pred={'example_id':ex.example_id,'replicate':0,'raw_text':'malformed'}
    assert score([request],targets,[pred])['replicates'][0]['correct']==0
    pred['raw_text']='{"value":null}'
    assert score([request],targets,[pred])['replicates'][0]['correct']==1
    with pytest.raises(ValueError,match='duplicate'):score([request],targets,[pred,pred])
    pred['replicate']=1
    with pytest.raises(ValueError,match='replicate'):score([request],targets,[pred])

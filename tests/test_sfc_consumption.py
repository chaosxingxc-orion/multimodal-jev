import copy
import hashlib
import json
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc import parse_calls,source_identity
from multimodal_jev.sfc_consumption import split_record,restore_record,export,SPLITS,validate_request


def record(split,stage):
    paths=[f'audio/final_data_audio/level_3_1/{split}/id_1_{i}.wav' for i in range(1,stage+1)]
    turns=[{'role':'user','content':'functions <audio>'},{'role':'assistant','content':'official history'},
           {'role':'user','content':'supplement <audio>'}][:2*stage-1]
    solution=json.dumps(['f(x=["a","b"],missing="NAN")','f(x=["a","b"],missing="NAN")'])
    return dict(example_id='sfc:'+paths[-1],input={'messages':turns,'audios':['/dataset/'+p for p in paths]},
        supervision={'official_solution':solution,'calls':parse_calls(solution)},
        metadata={**source_identity(paths),'relative_audios':paths,
            'audio_sha256':[hashlib.sha256(p.encode()).hexdigest() for p in paths]})


def source(tmp_path):
    root=tmp_path/'source';root.mkdir();pins={}
    for split in SPLITS:
        path=root/(split+'.jsonl');path.write_bytes(b''.join(canonical(record(split,i)) for i in (1,2)))
        pins[split]=dict(records=2,source_groups=1,sha256=file_hash(path))
    (root/'MANIFEST.json').write_bytes(canonical(dict(protocol='sfc-official-v1',splits=pins)))
    return root


def test_lossless_native_roundtrip_preserves_arrays_duplicates_nan_and_history():
    row=record('test_id',2);request,target=split_record(row)
    assert restore_record(request,target)==row
    assert set(request)=={'id','split','source_group','input','audio_sha256'}
    assert 'official_solution' not in json.dumps(request) and 'supervision' not in json.dumps(request)
    assert request['input']['messages'][1]['content']=='official history'
    assert len(target['supervision']['calls'])==2
    altered=copy.deepcopy(request);altered['input']['audios'].reverse()
    with pytest.raises(ValueError,match='binding'):restore_record(altered,target)
    with pytest.raises(ValueError,match='only request'):validate_request({**request,'supervision':{}})


def test_full_split_export_and_immutable_release(tmp_path):
    root=source(tmp_path);path,report=export(root,tmp_path/'out')
    for split in SPLITS:
        req=list(map(json.loads,(path/(split+'.requests.jsonl')).read_text().splitlines()))
        gold=list(map(json.loads,(path/(split+'.targets.jsonl')).read_text().splitlines()))
        original=list(map(json.loads,(root/(split+'.jsonl')).read_text().splitlines()))
        assert [restore_record(a,b) for a,b in zip(req,gold)]==original
        assert report['splits'][split]['two_audio_requests']==1
    with pytest.raises(FileExistsError):export(root,tmp_path/'out')


@pytest.mark.parametrize('change',['missing_stage','audio_prefix','audio_order','split_leak','target'])
def test_loss_or_custody_errors_rejected(tmp_path,change):
    root=source(tmp_path);path=root/'train_id.jsonl';rows=list(map(json.loads,path.read_text().splitlines()))
    if change=='missing_stage':rows.pop()
    if change=='audio_prefix':rows[1]['metadata']['audio_sha256'][0]='a'*64
    if change=='audio_order':rows[1]['metadata']['relative_audios'].reverse()
    if change=='split_leak':rows[0]['metadata']['audio_sha256'][0]=record('test_id',1)['metadata']['audio_sha256'][0];rows[1]['metadata']['audio_sha256'][0]=rows[0]['metadata']['audio_sha256'][0]
    if change=='target':rows[0]['supervision']['calls']=[]
    path.write_bytes(b''.join(map(canonical,rows)))
    manifest=json.loads((root/'MANIFEST.json').read_text());manifest['splits']['train_id']['sha256']=file_hash(path)
    (root/'MANIFEST.json').write_bytes(canonical(manifest))
    with pytest.raises(ValueError):export(root,tmp_path/'out')
    assert not list((tmp_path/'out').glob('.sfc-native-*'))


def test_stage_specific_function_order_is_preserved_not_rewritten(tmp_path):
    root=source(tmp_path);path=root/'train_id.jsonl';rows=list(map(json.loads,path.read_text().splitlines()))
    rows[1]['input']['messages'][0]['content']='functions in different order <audio>'
    path.write_bytes(b''.join(map(canonical,rows)))
    manifest=json.loads((root/'MANIFEST.json').read_text());manifest['splits']['train_id']['sha256']=file_hash(path)
    (root/'MANIFEST.json').write_bytes(canonical(manifest))
    release,receipt=export(root,tmp_path/'out')
    requests=list(map(json.loads,(release/'train_id.requests.jsonl').read_text().splitlines()))
    assert requests[1]['input']==rows[1]['input']
    assert receipt['splits']['train_id']['stage_specific_first_message_differences']==1

import copy
import hashlib
import json
import pytest
from multimodal_jev.consumption import canonical
from multimodal_jev.slurp_native import native_target, parse_prediction, accepts, messages, build, recording_index


def row(uid=1, split='train'):
    return {'slurp_id': uid, 'scenario': 'email', 'action': 'send', 'intent': 'send',
            'sentence': 'Amy Amy', 'tokens': [{'id': 0, 'surface': 'Amy'}, {'id': 1, 'surface': 'Amy'}],
            'entities': [{'type': 'person', 'span': [0]}, {'type': 'person', 'span': [1]}],
            'recordings': [{'file': split + str(uid) + '.flac'}]}


def test_all_entities_and_duplicate_fillers_preserved():
    target = native_target(row())
    assert target['entities'] == [{'type': 'person', 'filler': 'amy'}] * 2
    assert parse_prediction(json.dumps(target)) == target
    bad = row(); bad['entities'][0]['span'] = [99]
    with pytest.raises(ValueError, match='span'): native_target(bad)


@pytest.mark.parametrize('raw', ['{"scenario":"a","action":"b","entities":{}}',
    '{"scenario":"a","action":"b","entities":[{"type":"x","filler":""}]}',
    '{"scenario":"a","action":"b","entities":[],"entities":[]}',
    '{"scenario":"a","action":"b","entities":[],"extra":0}', 'invalid'])
def test_malformed_composites_rejected(raw):
    with pytest.raises(ValueError): parse_prediction(raw)


def test_native_export_keeps_every_split_and_labels_separate(tmp_path):
    root=tmp_path/'input';(root/'metadata').mkdir(parents=True);(root/'slurp_real').mkdir()
    state=tmp_path/'state';state.mkdir()
    metadata={}
    for uid, split in enumerate(['train', 'devel', 'test']):
        sample=row(uid,split)
        (root/'metadata'/(split+'.jsonl')).write_bytes(canonical(sample))
        (root/'slurp_real'/sample['recordings'][0]['file']).write_bytes(('audio'+split).encode())
        metadata[str(uid)]={'recordings':{sample['recordings'][0]['file']:{'usrid':'speaker'+str(uid)}}}
    (root/'metadata/metadata.json').write_text(json.dumps(metadata))
    output, receipt=build(root,tmp_path/'releases',state)
    assert [v['recordings'] for v in receipt['splits'].values()]==[1,1,1]
    assert receipt['splits']['train']['entity_occurrences_recording_weighted']==2
    request=json.loads((output/'train.requests.jsonl').read_text())
    target=json.loads((output/'train.targets.jsonl').read_text())
    assert target['metadata']['raw_intent']=='send'
    taxonomy=json.loads((output/'taxonomy.json').read_text())
    prompt=messages(request,taxonomy,'actual transcript','decide')
    assert 'Amy' not in json.dumps(prompt) and request['id'] not in json.dumps(prompt)
    assert 'target' not in request and 'transcript_reference' not in json.dumps(request)
    with pytest.raises(FileExistsError):build(root,tmp_path/'releases',state)


def test_duplicate_file_retains_official_last_label_and_links_all_utterances():
    first=row(1);second=row(2);third=row(3)
    first['recordings']=[{'file':'shared.flac'},{'file':'first.flac'}]
    second['recordings']=[{'file':'shared.flac'},{'file':'second.flac'}]
    second['tokens'][0]['surface']='Bob'
    third['recordings']=[{'file':'second.flac'},{'file':'third.flac'}]
    owners,groups,members,audit=recording_index({'train':[first,second,third]})
    assert owners['shared.flac']['row_index']==1
    assert owners['shared.flac']['target']['entities'][0]['filler']=='bob'
    assert len({groups['slurp:'+str(i)] for i in (1,2,3)})==1
    assert len(next(iter(members.values())))==3
    assert len(audit)==2 and audit[0]['conflicting_targets']


def test_duplicate_file_must_not_cross_official_split():
    first=row(1);second=row(2,'test');second['recordings']=first['recordings']
    with pytest.raises(ValueError,match='crosses'):
        recording_index({'train':[first],'test':[second]})


def test_export_resolves_duplicate_once_and_retains_audit(tmp_path):
    root=tmp_path/'input';(root/'metadata').mkdir(parents=True);(root/'slurp_real').mkdir()
    state=tmp_path/'state';state.mkdir();metadata={}
    first=row(1);second=row(2)
    first['recordings']=[{'file':'shared.flac'},{'file':'only-first.flac'}]
    second['recordings']=[{'file':'shared.flac'},{'file':'only-second.flac'}]
    second['tokens'][0]['surface']='Bob'
    for split, samples in [('train',[first,second]),('devel',[row(3,'devel')]),('test',[row(4,'test')])]:
        (root/'metadata'/(split+'.jsonl')).write_bytes(b''.join(map(canonical,samples)))
        for sample in samples:
            files={r['file']:{'usrid':'same-speaker'} for r in sample['recordings']}
            metadata[str(sample['slurp_id'])]={'recordings':files}
            for filename in files:(root/'slurp_real'/filename).write_bytes(filename.encode())
    (root/'metadata/metadata.json').write_text(json.dumps(metadata))
    output,receipt=build(root,tmp_path/'releases',state)
    requests=list(map(json.loads,(output/'train.requests.jsonl').read_text().splitlines()))
    targets=list(map(json.loads,(output/'train.targets.jsonl').read_text().splitlines()))
    assert len(requests)==3 and len({r['source_group'] for r in requests})==1
    target=next(t for t in targets if t['metadata']['file']=='shared.flac')
    assert target['metadata']['slurp_id']==2 and target['target']['entities'][0]['filler']=='bob'
    assert receipt['splits']['train']['raw_recording_entries']==4
    assert receipt['splits']['train']['recordings']==3
    assert receipt['duplicate_recording_audit'][0]['conflicting_targets']

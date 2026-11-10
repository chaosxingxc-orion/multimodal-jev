import copy
import hashlib
import json
import pytest
from multimodal_jev.consumption import canonical
from multimodal_jev.slurp_native import native_target, parse_prediction, accepts, messages, build


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

import copy
import hashlib
import json

import pytest

from multimodal_jev.consumption import canonical
from multimodal_jev.slurp_clean_split import partition
from multimodal_jev.slurp_clean_sources import assemble, publish
from multimodal_jev.baseline import file_hash


def fixture():
    rows = [dict(slurp_id=i, sentence='sentence '+str(i), intent=intent,
                 scenario='email', action='query',
                 tokens=[dict(id=0, surface=name)],
                 entities=[dict(type='person', span=[0]), dict(type='person', span=[0])],
                 recordings=[dict(file=file)])
            for i, intent, name, file in [(1,'email_query','Amy','a.flac'),
                                         (2,'query','Bob','b.flac'),(3,'query','Eve','b.flac')]]
    requests = [dict(id='slurp:'+f, split='train', source_group='old-group',
                     audio_sha256=h*64, input=dict(audio='/audio/'+f, schema_sha256='s'*64))
                for f,h in [('a.flac','a'),('b.flac','b')]]
    return rows, requests, partition(rows,requests)


def test_conflict_mask_multiplicity_and_label_free_input():
    rows, requests, split = fixture()
    bundles, targets, quarantine, taxonomy = assemble(rows, requests, split)
    assert [x['id'] for x in bundles] == ['slurp:a.flac']
    assert [x['id'] for x in quarantine] == ['slurp:b.flac']
    assert bundles[0]['input'] == {'audio':'/audio/a.flac'}
    assert targets[0]['native_target']['entities'] == [dict(type='person',filler='amy')]*2
    assert targets[0]['request_sha256'] == hashlib.sha256(canonical(bundles[0])).hexdigest()
    assert targets[0]['optimization_allowed'] is False
    assert taxonomy['intent_alias_map'] == [dict(scenario='email',action='query',raw_names=['email_query','query'])]
    assert 'sentence' not in json.dumps(bundles+targets)


@pytest.mark.parametrize('mutation', ['role', 'conflict', 'missing', 'leak', 'path'])
def test_reject_changed_partition_or_request_contract(mutation):
    rows, requests, split = fixture()
    if mutation=='role': split['records'][0]['role']='test'
    if mutation=='conflict': split['records'][1]['supervision_status']='available'
    if mutation=='missing': split['records'].pop()
    if mutation=='leak': requests[0]['input']['transcript']='secret'
    if mutation=='path': requests[0]['input']['audio']='/audio/wrong.flac'
    with pytest.raises(ValueError): assemble(rows,requests,split)


def test_publisher_authentication_and_immutable_output(tmp_path):
    rows, requests, split = fixture()
    clean=tmp_path/'clean'; native=tmp_path/'native'; root=tmp_path/'split'
    (clean/'metadata').mkdir(parents=True);native.mkdir();root.mkdir()
    (clean/'metadata/train.jsonl').write_bytes(b''.join(canonical(r) for r in rows))
    (clean/'ALIGNMENT-RECEIPT.json').write_text('{}')
    (native/'MANIFEST.json').write_text('{}')
    (native/'train.requests.jsonl').write_bytes(b''.join(canonical(r) for r in requests))
    pins={k:file_hash(p) for k,p in {
        'clean_metadata_sha256':clean/'metadata/train.jsonl',
        'alignment_sha256':clean/'ALIGNMENT-RECEIPT.json',
        'native_manifest_sha256':native/'MANIFEST.json',
        'requests_sha256':native/'train.requests.jsonl'}.items()}
    split['identity']=pins
    (root/'source-split.json').write_bytes(canonical(split))
    counts={r:{'available':sum(x['role']==r and x['supervision_status']=='available' for x in split['records'])}
            for r in ('learning','selection')}
    evidence=tmp_path/'verified.json'
    evidence.write_bytes(canonical(dict(identity=pins,split_sha256=file_hash(root/'source-split.json'),
                                       recordings=2,quarantined=1,counts=counts)))
    out=tmp_path/'release'
    result=publish(clean,native,root,evidence,out)
    assert result['optimization_ready'] is False
    assert (out/'learning.targets.jsonl').stat().st_mode & 0o777 == 0o600
    assert sum(result['counts'].values())==1
    with pytest.raises(FileExistsError): publish(clean,native,root,evidence,out)
    (native/'train.requests.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='authenticated input changed'):
        publish(clean,native,root,evidence,tmp_path/'new-release')
    assert not (tmp_path/'new-release').exists()

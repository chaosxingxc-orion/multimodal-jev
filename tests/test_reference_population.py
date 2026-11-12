import copy
import pytest
from multimodal_jev.reference_population import compare


def fixture():
    roster=[dict(id='a',audio_sha256='same',gold='y'),dict(id='b',audio_sha256='different',gold='x')]
    jobs=[dict(sample_id='b',audio_sha256='same',audio={'audio_sha256':'same'},ranking=[{'index':1,'label':'x'}])]
    return roster,jobs,['x','y']


def test_hash_not_id_or_filename_matches_and_candidates_stay_distinct():
    r=compare(*fixture());assert r['exact_audio_matched_recordings']==1
    assert r['matches'][0]['id']=='a' and r['unmatched_ids']==['b']
    assert r['matched_full_candidate_recordings']==0
    assert not r['matches'][0]['reference_requests'][0]['reviewed_gold_in_candidates']


@pytest.mark.parametrize('bad',['duplicate_id','audio','indices','labels','outside'])
def test_reference_identity_errors_rejected(bad):
    r,j,o=fixture()
    if bad=='duplicate_id':j.append(copy.deepcopy(j[0]))
    elif bad=='audio':j[0]['audio']['audio_sha256']='changed'
    elif bad=='indices':j[0]['ranking'][0]['index']=2
    elif bad=='labels':j[0]['ranking'].append({'index':2,'label':'x'})
    else:j[0]['ranking'][0]['label']='unknown'
    with pytest.raises(ValueError):compare(r,j,o)

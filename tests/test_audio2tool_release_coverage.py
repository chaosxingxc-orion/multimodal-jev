from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from audit_audio2tool_release_coverage import coverage


def fixture():
    path='tier4_implicit_audios/query_00001/speaker_01_x.wav'
    rows=[dict(id=i,query_idx=10,path=path,expected_tool_call='PRIVATE',reasoning=str(i)) for i in (1,2)]
    inv=[dict(tier=4,audios=[dict(path='public/'+path,sha256='a'*64)])]
    return rows,inv


def test_repeated_path_keeps_both_release_records_and_hides_values():
    rows,inv=fixture();mapping,report=coverage(rows,4,inv)
    assert len(mapping)==2 and report['unique_audio_paths']==1
    assert report['duplicate_paths'][0]['differing_fields']==['id','reasoning']
    assert 'PRIVATE' not in str((mapping,report))


@pytest.mark.parametrize('fault',['duplicate_id','unknown_path','omitted_audio','traversal'])
def test_coverage_failures_are_not_silently_dropped(fault):
    rows,inv=fixture()
    if fault=='duplicate_id':rows[1]['id']=1
    elif fault=='unknown_path':rows[1]['path']=rows[1]['path'].replace('query_00001','query_00002')
    elif fault=='omitted_audio':inv[0]['audios'].append(dict(path='public/extra.wav',sha256='b'*64))
    else:rows[1]['path']='../x/y.wav'
    with pytest.raises(ValueError):coverage(rows,4,inv)

from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from audit_audio2tool_dialogue_protocol import project


def fixture(roles):
    turns=[dict(role=role,path=f'tier7_multiturn_audios/query_00001/turn_{i:02d}_{role}_x.wav',
                content='synthetic') for i,role in enumerate(roles)]
    row=dict(id=1,query_idx=2100,chat_history=turns,user_request='synthetic',expected_tool_call='SECRET_TARGET')
    index=dict(query_id='query_00001',audios=[dict(path='public/'+t['path'],role=t['role'],turn=i)
                                            for i,t in enumerate(turns)])
    return row,index


def test_trailing_agent_is_preserved_but_not_mistaken_for_final_request():
    row,index=fixture(['user','agent','user','agent'])
    result=project(row,index)
    assert result['full_dialogue_audio_count']==4
    assert result['decision_prefix_audio_count']==3
    assert result['post_final_user_audio_count']==1
    assert result['directory_matches_release_id']
    assert not result['directory_matches_source_query_idx']
    assert 'SECRET_TARGET' not in str(result) and 'synthetic' not in str(result)


def test_consecutive_users_are_reported_not_deleted():
    row,index=fixture(['user','agent','user','user'])
    result=project(row,index)
    assert result['adjacent_same_role_positions']==[3]
    assert result['decision_prefix_audio_count']==4


@pytest.mark.parametrize('fault',['order','role','turn','path'])
def test_incompatible_inventory_rejected(fault):
    row,index=fixture(['user','agent','user'])
    if fault=='order':index['audios'].reverse()
    elif fault=='role':index['audios'][1]['role']='user'
    elif fault=='turn':index['audios'][1]['turn']=7
    else:row['chat_history'][0]['path']='../query_00001/x.wav'
    with pytest.raises(ValueError):project(row,index)

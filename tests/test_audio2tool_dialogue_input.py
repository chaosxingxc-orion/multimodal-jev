import copy
import pytest

from multimodal_jev.audio2tool_dialogue_input import decision_input, messages

PIN='a'*64


def inventory(roles):
    return dict(tier=7,query_id='query_00001',audio_semantics='ordered_conversation_turns',
        audios=[dict(path=f'public/tier7_multiturn_audios/query_00001/turn_{i:02d}_{r}_x.wav',
                     sha256='b'*64,bytes=100,turn=i,role=r) for i,r in enumerate(roles)])


def test_future_audio_kept_only_in_provenance_and_waveform_order_preserved():
    row=inventory(['user','agent','user','agent']);before=copy.deepcopy(row)
    result=decision_input(row,PIN)
    assert row==before
    assert result['provenance']['post_decision_audios']==row['audios'][3:]
    waves=[object() for _ in range(3)]
    rendered=messages(result['input'],waves,'independent task',PIN)
    assert [x['audio'] for x in rendered[1]['content'] if x['type']=='audio']==waves
    assert len(rendered[1]['content'])==6
    assert rendered[1]['content'][2]['text']=='Conversation turn 1, speaker role agent:'
    with pytest.raises(ValueError):messages(result['input'],waves+[object()],'task',PIN)


def test_consecutive_user_turns_remain_distinct():
    result=decision_input(inventory(['user','agent','user','user']),PIN)
    assert len(result['input']['audios'])==4
    assert result['provenance']['post_decision_audios']==[]


@pytest.mark.parametrize('field',['query','user_request','reasoning','functions','expected_tool_call','content'])
def test_extra_annotation_fields_fail_closed(field):
    row=inventory(['user','agent','user']);row[field]='do not leak'
    with pytest.raises(ValueError):decision_input(row,PIN)
    row=inventory(['user','agent','user']);row['audios'][0][field]='do not leak'
    with pytest.raises(ValueError):decision_input(row,PIN)
    result=decision_input(inventory(['user','agent','user']),PIN)
    result['input'][field]='do not leak'
    with pytest.raises(ValueError):messages(result['input'],[1,2,3],'task',PIN)


@pytest.mark.parametrize('fault',['wrong_registry','duplicate_turn','traversal','wrong_role','missing_user'])
def test_invalid_identity_or_boundary_rejected(fault):
    row=inventory(['user','agent','user'])
    if fault=='duplicate_turn':row['audios'][1]['turn']=0
    elif fault=='traversal':row['audios'][1]['path']='../other.wav'
    elif fault=='wrong_role':row['audios'][1]['role']='user'
    elif fault=='missing_user':row=inventory(['agent','agent'])
    if fault=='wrong_registry':
        result=decision_input(row,PIN)
        with pytest.raises(ValueError):messages(result['input'],[1,2,3],'task','c'*64)
    else:
        with pytest.raises(ValueError):decision_input(row,PIN)

import pytest
from multimodal_jev.omni_runtime import task_messages, validate_thinker_loading


def test_audio_and_dynamic_definition_share_user_turn():
    wave=object();messages=task_messages(wave,'{"question":"Is it on?"}')
    assert messages[1]['content'][0]['audio'] is wave
    assert messages[1]['content'][1]['text']=='{"question":"Is it on?"}'
    assert len(messages)==2


def test_only_official_non_thinker_tensors_may_be_unused():
    index=['thinker.model.weight','thinker.audio_tower.weight','talker.weight','token2wav.weight']
    state=['model.weight','audio_tower.weight']
    validate_thinker_loading({'unexpected_keys':['talker.weight','token2wav.weight']},index,state)
    for loading in [{'missing_keys':['audio_tower.weight']},{'mismatched_keys':['model.weight']},
                    {'unexpected_keys':['thinker.audio_tower.weight']},
                    {'unexpected_keys':['talker.not_in_checkpoint']}]:
        with pytest.raises(ValueError):validate_thinker_loading(loading,index,state)
    with pytest.raises(ValueError):validate_thinker_loading({},index,['model.weight'])

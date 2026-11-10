import numpy as np
import pytest
from multimodal_jev.waveform_control import condition_audio, answer_identity, transitions


def test_intervention_preserves_duration_and_original_waveform():
    wave=np.array([1,-.5,0],dtype=np.float32)
    assert condition_audio(wave,'actual','test') is wave
    zeros=condition_audio(wave,'zeros','train')
    assert zeros.shape==wave.shape and zeros.dtype==wave.dtype
    assert not zeros.any() and wave[0]==1
    for condition,split in [('zeros','test'),('zeros','valid'),('transcript','train')]:
        with pytest.raises(ValueError):condition_audio(wave,condition,split)


def row(raw):
    return dict(raw_text=raw,request_sha256='request',prompt_sha256='prompt',
                source_group='group',split='train',input_tokens=5,replicate=0)


def test_invalid_null_boolean_and_integer_are_distinct():
    assert answer_identity('{"value":null}')!=answer_identity('null')
    assert answer_identity('{"value":true}')!=answer_identity('{"value":1}')
    a={'a':row('invalid A'),'b':row('{"value":null}'),'c':row('{"value":true}')}
    z={'a':row('invalid B'),'b':row('invalid'),'c':row('{"value":1}')}
    report=transitions(a,z)
    assert report['changed_raw']==3 and report['changed_decisions']==2
    assert report['both_invalid']==1 and report['actual_only_valid']==1
    with pytest.raises(ValueError):transitions(a,{'a':z['a']})
    z['c']['input_tokens']=6
    with pytest.raises(ValueError,match='input_tokens'):transitions(a,z)

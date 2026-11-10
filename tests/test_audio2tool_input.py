import copy
from pathlib import Path
import sys
import pytest
from multimodal_jev import audio2tool_input as api
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from prepare_audio2tool_inputs import build


def sample(tier=4):
    path=(f'public/tier{tier}_implicit_audios/query_00001/speaker_01_x.wav' if tier!=8 else
          'public/tier8_intent_blending_audios/query_00001/mixed.wav')
    return dict(mode='foreground_mixture' if tier==8 else 'single_audio',registry_sha256=api.REGISTRY_SHA256,
                audios=[dict(path=path,sha256='b'*64,bytes=100)])


def test_duplicate_release_records_preserved():
    inp=sample();a=inp['audios'][0]
    mapping=[dict(tier=4,release_id=i,source_query_idx=10,audio_path=a['path'],audio_sha256=a['sha256']) for i in (1,2)]
    result=build([dict(tier=4,audios=[a])],mapping,[],[])
    assert len(result)==2 and result[0]['id']!=result[1]['id']
    assert result[0]['input']==result[1]['input']


def test_mixture_uses_original_waveform_with_foreground_instruction(monkeypatch):
    monkeypatch.setattr(api,'system_text',lambda data:'fixed tool definitions')
    inp=sample(8);wave=object();rendered=api.messages(inp,[wave],b'registry')
    assert rendered[1]['content'][1]['audio'] is wave
    assert 'foreground' in rendered[1]['content'][0]['text']
    assert 'background_query' not in str(rendered)


@pytest.mark.parametrize('field',['query','functions','expected_tool_call','background_query','reasoning'])
def test_unexpected_annotation_fields_rejected(field):
    inp=sample();inp[field]='secret'
    with pytest.raises(ValueError):api.validate(inp)
    inp=sample();inp['audios'][0][field]='secret'
    with pytest.raises(ValueError):api.validate(inp)


def test_tier8_mode_mismatch_and_missing_audio_rejected():
    inp=sample(8);inp['mode']='single_audio'
    with pytest.raises(ValueError):api.validate(inp)
    with pytest.raises(ValueError):api.messages(sample(),[],b'bad')


def test_registry_bytes_cannot_be_substituted():
    with pytest.raises(ValueError,match='registry hash'):api.system_text(b'unknown registry')

import copy
import hashlib
from io import BytesIO

import numpy as np
import pytest
import soundfile as sf

from multimodal_jev.audio2tool_audio import load_audio
from multimodal_jev.audio2tool_input import REGISTRY_SHA256


def asset(root, *, mode='single_audio', samples=241, rate=24000, role=None, turn=0):
    if role:
        rel=f'public/tier7_multiturn_audios/query_00001/turn_{turn:02d}_{role}_x.wav'
    elif mode=='foreground_mixture':
        rel='public/tier8_intent_blending_audios/query_00001/mixed.wav'
    else:
        rel='public/tier1_direct_audios/query_00001/speaker_01_x.wav'
    buffer=BytesIO()
    sf.write(buffer,np.linspace(-.2,.2,samples,dtype=np.float32),rate,format='WAV',subtype='PCM_16')
    payload=buffer.getvalue();path=root/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
    a=dict(path=rel,sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
    if role:a.update(role=role,turn=turn)
    return dict(mode=mode,registry_sha256=REGISTRY_SHA256,audios=[a])


@pytest.mark.parametrize('mode',['single_audio','foreground_mixture'])
def test_real_decode_resampling_and_only_right_padding(tmp_path,mode):
    from scipy.signal import resample_poly
    inp=asset(tmp_path,mode=mode)
    original,_=sf.read(tmp_path/inp['audios'][0]['path'],dtype='float32')
    expected=resample_poly(original,2,3)
    waves,receipts=load_audio(inp,tmp_path)
    assert len(expected)==161 and len(waves[0])==320
    np.testing.assert_array_equal(waves[0][:161],expected)
    assert np.count_nonzero(waves[0][161:])==0
    assert receipts[0]['right_zero_padding_samples']==159
    assert receipts[0]['expected_frames']==2


def test_full_dialogue_order_including_agent_audio(tmp_path):
    inp=asset(tmp_path,mode='dialogue_last_user',role='user',turn=0,samples=240)
    for turn,role in [(1,'agent'),(2,'user')]:
        inp['audios']+=asset(tmp_path,mode='dialogue_last_user',role=role,turn=turn,samples=240*(turn+1))['audios']
    waves,receipts=load_audio(inp,tmp_path)
    assert [len(w) for w in waves]==[160,320,480]
    assert [r['path'] for r in receipts]==[a['path'] for a in inp['audios']]
    swapped=copy.deepcopy(inp);swapped['audios'].reverse()
    with pytest.raises(ValueError):load_audio(swapped,tmp_path)


@pytest.mark.parametrize('mutation',['hash','size','format','escape'])
def test_source_mismatch_rejected(tmp_path,mutation):
    inp=asset(tmp_path,rate=16000 if mutation=='format' else 24000)
    a=inp['audios'][0]
    if mutation=='hash':a['sha256']='0'*64
    if mutation=='size':a['bytes']+=1
    if mutation=='escape':
        path=tmp_path/a['path'];path.unlink();path.symlink_to('/etc/hosts')
    with pytest.raises(ValueError):load_audio(inp,tmp_path)


def test_request_binds_verified_waves_and_rejects_lost_placeholder(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from multimodal_jev import audio2tool_input
    from multimodal_jev.audio2tool_audio import prepare_request
    monkeypatch.setattr(audio2tool_input,'system_text',lambda _: 'fixed registry fixture')
    inp=asset(tmp_path)
    rendered=[]
    def template(msgs,**kwargs):
        rendered.append(msgs)
        assert kwargs==dict(tokenize=False,add_generation_prompt=True)
        return '<|audio_pad|>'
    processor=SimpleNamespace(feature_extractor=SimpleNamespace(hop_length=160),apply_chat_template=template)
    request,receipts=prepare_request(processor,inp,tmp_path,b'fixture')
    wave,rate=request['multi_modal_data']['audio'][0]
    assert rate==16000 and wave is rendered[0][1]['content'][1]['audio']
    assert receipts[0]['actor_samples']==len(wave)
    processor.apply_chat_template=lambda *args,**kw:'no audio'
    with pytest.raises(ValueError,match='placeholders'):prepare_request(processor,inp,tmp_path,b'fixture')


def test_bad_registry_rejected_before_audio_read(tmp_path):
    from types import SimpleNamespace
    from multimodal_jev.audio2tool_audio import prepare_request
    inp=asset(tmp_path)
    (tmp_path/inp['audios'][0]['path']).unlink()
    with pytest.raises(ValueError,match='registry hash'):
        prepare_request(SimpleNamespace(),inp,tmp_path,b'wrong registry')

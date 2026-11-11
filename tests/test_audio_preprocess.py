import numpy as np
import pytest
from multimodal_jev.audio_preprocess import AudioPreprocess,preprocess_audio


def test_identity_is_exact_nonmutating_and_hash_stable():
    x=np.array([0.,.1,-.3,1.1],dtype=np.float32);copy=x.copy()
    y,r=preprocess_audio(x)
    assert y.tobytes()==x.tobytes() and r['input_waveform_sha256']==r['output_waveform_sha256']
    y[0]=1
    np.testing.assert_array_equal(x,copy)


def test_edge_trim_preserves_internal_pause_and_silent_inputs():
    x=np.r_[np.zeros(3200),np.ones(1600)*.1,np.zeros(3200),np.ones(1600)*.1,np.zeros(3200)].astype('float32')
    y,r=preprocess_audio(x,dict(trim_threshold_dbfs=-40,trim_margin_ms=0))
    assert r['retained_original_interval']==[3200,9600]
    np.testing.assert_array_equal(y,x[3200:9600])
    silence=np.zeros(16000,dtype='float32')
    y,_=preprocess_audio(silence,dict(trim_threshold_dbfs=-40,target_rms_dbfs=-20))
    np.testing.assert_array_equal(y,silence)


def test_gain_limit_and_preemphasis_equation():
    x=np.ones(1600,dtype='float32')*.001
    y,r=preprocess_audio(x,dict(target_rms_dbfs=-10,max_gain_db=6))
    assert r['applied_gain']==pytest.approx(10**(.3)) and np.max(y)<.0021
    x=np.array([.1,.2,.3],dtype='float32')
    y,_=preprocess_audio(x,dict(preemphasis=.97))
    np.testing.assert_allclose(y,[.1,.103,.106],atol=1e-7)
    y,_=preprocess_audio(np.array([.99,-.99],dtype='float32'),dict(preemphasis=.97))
    assert np.max(np.abs(y))<=.990001


def test_speed_duration_and_pitch_semantics_are_explicit():
    pytest.importorskip('scipy')
    x=(.1*np.sin(2*np.pi*400*np.arange(16000)/16000)).astype('float32')
    y,r=preprocess_audio(x,dict(speed_factor=1.25))
    assert len(y)==12800 and r['speed_changes_pitch']
    peak=np.argmax(np.abs(np.fft.rfft(y)))*16000/len(y)
    assert peak==pytest.approx(500,abs=2)


@pytest.mark.parametrize('options',[dict(speed_factor=0),dict(preemphasis=1),dict(target_rms_dbfs=float('nan')),dict(trim_margin_ms=-1)])
def test_invalid_processing_rejected(options):
    with pytest.raises(ValueError):AudioPreprocess(**options)

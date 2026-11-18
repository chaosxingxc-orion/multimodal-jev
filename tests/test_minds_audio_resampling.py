import numpy as np
import pytest
from tools.reports.resample_minds_context_inputs import convert


def test_duration_and_tone_frequency_preserved():
    x=np.sin(2*np.pi*440*np.arange(8000)/8000).astype(np.float32)
    y=convert(x,8000)
    assert len(y)==16000
    assert np.argmax(np.abs(np.fft.rfft(y)))==440
    assert abs(np.sqrt(np.mean(y[200:-200]**2))-np.sqrt(.5))<.002


def test_impulse_alignment_and_no_amplitude_clipping():
    x=np.zeros(101,dtype=np.float32);x[50]=2
    y=convert(x,8000)
    assert np.argmax(y)==100
    assert y.max()>1


@pytest.mark.parametrize('x,rate',[(np.zeros(1,dtype=np.float32),16000),
    (np.array([np.nan],dtype=np.float32),8000),(np.zeros((5,2),dtype=np.float32),8000),
    (np.array([],dtype=np.float32),8000)])
def test_invalid_source_rejected(x,rate):
    with pytest.raises(ValueError):convert(x,rate)

import numpy as np
import pytest
from multimodal_jev.omni_waveform_boundary import align_to_hop


def test_padding_preserves_all_samples_order_and_is_idempotent():
    waves=[np.arange(n,dtype=np.float32)+i for i,n in enumerate([159,160,161,321])]
    copies=[w.copy() for w in waves];result=align_to_hop(waves,160)
    assert [len(w) for w in result]==[160,160,320,480]
    for source,before,after in zip(waves,copies,result):
        np.testing.assert_array_equal(source,before)
        np.testing.assert_array_equal(after[:len(source)],source)
        assert not after[len(source):].any()
    for left,right in zip(result,align_to_hop(result,160)):
        np.testing.assert_array_equal(left,right)


@pytest.mark.parametrize('wave',[np.array([],dtype=np.float32),np.zeros((2,3),dtype=np.float32),
    np.ones(3,dtype=np.float64),np.array([np.nan],dtype=np.float32)])
def test_invalid_waveform_rejected(wave):
    with pytest.raises(ValueError):align_to_hop([wave],160)


@pytest.mark.parametrize('hop',[0,-1,True,1.5])
def test_invalid_hop_rejected(hop):
    with pytest.raises(ValueError):align_to_hop([np.ones(3,dtype=np.float32)],hop)

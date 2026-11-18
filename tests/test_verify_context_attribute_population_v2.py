import copy
import hashlib
from pathlib import Path
import pytest
import torch

from tools.reports.verify_context_attribute_population_v2 import frame_geometry,validate_audio_masks,derive


@pytest.mark.parametrize('samples',[1,7999,8000,8191,16000,27648,114672,373760,16001])
def test_exact_single_waveform_mask_and_convolution_geometry(samples):
    frames,width,tokens=frame_geometry(samples)
    # Independent downsampling by three stride-2 convolutions per chunk.
    chunks=[min(100,frames-i) for i in range(0,frames,100)]
    expected=0
    for n in chunks:
        for _ in range(3):n=(n-1)//2+1
        expected+=n
    assert tokens==expected
    mask=(torch.arange(width)<frames).to(torch.int32)[None]
    row=dict(samples=samples,records=[dict(audio_tokens=tokens,tensors={
        'input_features_mask':dict(shape=list(mask.shape),dtype=str(mask.dtype),sha256=hashlib.sha256(mask.numpy().tobytes()).hexdigest()),
        'input_features':dict(shape=[1,128,width],dtype='torch.float32')})])
    assert list(validate_audio_masks([row]))==[row]
    bad=copy.deepcopy(row);bad['records'][0]['tensors']['input_features_mask']['sha256']='0'*64
    with pytest.raises(ValueError):list(validate_audio_masks([bad]))


def test_v1_preserved_and_both_population_adaptations_available():
    for role in ['learning','selection']:
        module=derive(Path.cwd(),role)
        assert callable(module.verify)

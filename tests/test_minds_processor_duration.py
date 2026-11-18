import pytest
import torch
from tools.reports.verify_minds_processor_cache import validate_features


def features(frames,valid=None):
    length=((frames+99)//100)*100
    return dict(input_features=torch.zeros(1,128,length,dtype=torch.bfloat16),
        input_features_mask=(torch.arange(length)[None,:]<(frames if valid is None else valid)).long())


def test_long_request_not_clipped_to_thirty_seconds():
    assert validate_features(features(6790),6790*160)==6790
    with pytest.raises(ValueError):validate_features(features(3000),6790*160)


def test_padding_not_counted_as_audio_evidence():
    with pytest.raises(ValueError):validate_features(features(6800),6790*160)


def test_holey_mask_rejected_even_with_correct_sum():
    x=features(103);x['input_features_mask'][0,20]=0;x['input_features_mask'][0,120]=1
    with pytest.raises(ValueError):validate_features(x,103*160)


def test_endpoint_rounding_and_short_audio_minimum():
    assert validate_features(features(103),103*160+1)==103
    assert validate_features(features(104),103*160+1)==104
    assert validate_features(features(50),1000)==50
    with pytest.raises(ValueError):validate_features(features(102),103*160+1)

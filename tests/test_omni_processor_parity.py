from types import SimpleNamespace

import pytest
import torch

from multimodal_jev.omni_processor_parity import compare_prepared


def fixture():
    # Two ordered clips, different lengths, nonzero ignored padding. Values
    # intentionally require BF16 rounding at the model's input boundary.
    features = torch.tensor([[[.123456, .234567, .345678], [.4, .5, .6]],
                             [[.712345, .812345, 99.], [.9, 1., 98.]]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    item = dict(waveform_boundary='vllm_hop', input_ids=torch.tensor([5, 6, 7]),
                input_features=features, feature_attention_mask=mask)
    data = dict(input_audio_features=torch.cat((features[0], features[1, :, :2]), 1).bfloat16(),
                audio_feature_lengths=torch.tensor([3, 2]))
    processed = dict(prompt_token_ids=[5, 6, 7], mm_kwargs=SimpleNamespace(get_data=lambda: data))
    return item, processed, data


def test_actual_ordered_valid_features_at_model_dtype():
    item, processed, _ = fixture()
    result = compare_prepared(item, processed)
    assert result['ordered_frames'] == [3, 2]
    assert result['feature_shape'] == [2, 5]
    assert result['max_feature_error'] == 0


@pytest.mark.parametrize('change', [
    'tokens', 'lengths', 'order', 'truncation', 'dtype', 'nan', 'mask_hole', 'boundary',
])
def test_input_drift_rejected(change):
    item, processed, data = fixture()
    if change == 'tokens':
        processed['prompt_token_ids'] = [5, 7, 6]
    elif change == 'lengths':
        data['audio_feature_lengths'] = torch.tensor([2, 3])
    elif change == 'order':
        data['input_audio_features'] = data['input_audio_features'].flip(1)
    elif change == 'truncation':
        data['input_audio_features'] = data['input_audio_features'][:, :-1]
    elif change == 'dtype':
        data['input_audio_features'] = data['input_audio_features'].float()
    elif change == 'nan':
        data['input_audio_features'][0, 0] = float('nan')
    elif change == 'mask_hole':
        item['feature_attention_mask'][1] = torch.tensor([1, 0, 1])
    else:
        item['waveform_boundary'] = 'raw'
    with pytest.raises(ValueError):
        compare_prepared(item, processed)

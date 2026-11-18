import pytest
import torch

from tools.reports.replay_context_attention_capture import normalize_hf


def test_restore_token_axis_preserves_every_value_and_original_capture():
    original = torch.arange(30, dtype=torch.bfloat16).reshape(1, 30)
    record = {'layer_00.q_rope': [torch.zeros(5, 6)],
              'layer_00.kernel_output': [original]}
    result = normalize_hf(record, 5)
    assert result['layer_00.kernel_output'][0].shape == (5, 6)
    assert torch.equal(result['layer_00.kernel_output'][0].flatten(), original.flatten())
    assert record['layer_00.kernel_output'][0].shape == (1, 30)
    assert result['layer_00.q_rope'] is record['layer_00.q_rope']


def test_unknown_layout_or_wrong_length_is_rejected():
    record = {'layer_00.q_rope': [torch.zeros(5, 6)],
              'layer_00.kernel_output': [torch.zeros(1, 29)]}
    with pytest.raises(ValueError, match='known flattened'):
        normalize_hf(record, 5)
    with pytest.raises(ValueError, match='complete Q'):
        normalize_hf(record, 4)

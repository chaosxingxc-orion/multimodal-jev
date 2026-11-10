"""A stable candidate must still be rejected for changing original probabilities."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'tools/jev'))
spec = importlib.util.spec_from_file_location(
    'candidate_head_diagnostic', Path(__file__).parents[1]/'tools/jev/diagnose_omni_candidate_head.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_equal_selected_logits_do_not_hide_changed_normalization():
    original = torch.zeros(2, 4, dtype=torch.bfloat16)
    candidate = original.clone()
    candidate[:, 3] = 4
    report = module.compare_logits(candidate, original, [0, 1])
    assert not report['within_original_atol']
    assert report['response_logprob_max_error'] > 2
    assert module.compare_logits(candidate, candidate, [0, 1])['within_original_atol']


def test_constant_shift_is_probability_invariant_but_logits_differ():
    original = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.bfloat16)
    report = module.compare_logits(original+2, original, [1, 2])
    assert not report['logits_equal']
    assert report['logits_max_error'] == 2
    assert report['response_logprob_max_error'] == 0
    assert report['within_original_atol']


@pytest.mark.parametrize('candidate', [
    torch.zeros(2, 4),
    torch.zeros(1, 4, dtype=torch.bfloat16),
    torch.full((2, 4), float('nan'), dtype=torch.bfloat16),
])
def test_bad_logits_rejected(candidate):
    with pytest.raises(ValueError, match='matching finite'):
        module.compare_logits(candidate, torch.zeros(2, 4, dtype=torch.bfloat16), [0, 1])

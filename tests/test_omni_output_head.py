"""The output-head control must preserve GEMM shape and causal alignment."""
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    'output_head_diagnostic', Path(__file__).parents[1]/'tools/jev/diagnose_omni_output_head.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_full_sequence_projection_and_next_token_alignment():
    hidden = torch.arange(2*7*3).reshape(2, 7, 3).to(torch.bfloat16)
    calls = []

    def norm(x):
        calls.append(('norm', tuple(x.shape)))
        return x + 1

    def head(x):
        calls.append(('head', tuple(x.shape)))
        return torch.cat((x, x[..., :1]), dim=-1)

    actual = module.project_response(hidden, norm, head, 2)
    assert calls == [('norm', (2, 7, 3)), ('head', (2, 7, 3))]
    expected = torch.cat((hidden[:, 4:6]+1, hidden[:, 4:6, :1]+1), dim=-1)
    assert torch.equal(actual, expected)
    assert actual.untyped_storage().nbytes() == actual.numel()*actual.element_size()


@pytest.mark.parametrize('shape,dtype,length', [
    ((7, 3), torch.bfloat16, 2),
    ((1, 2, 3), torch.bfloat16, 2),
    ((1, 7, 3), torch.float32, 2),
    ((1, 7, 3), torch.bfloat16, 0),
])
def test_invalid_sequence_rejected_before_projection(shape, dtype, length):
    def forbidden(x):
        pytest.fail('invalid input reached projection')

    with pytest.raises(ValueError, match='complete BF16 sequence'):
        module.project_response(torch.zeros(shape, dtype=dtype), forbidden, forbidden, length)

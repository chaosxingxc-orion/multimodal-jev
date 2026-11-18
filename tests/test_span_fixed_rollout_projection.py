from types import SimpleNamespace

import pytest
import torch

from multimodal_jev.span_fixed_rollout_projection import (
    parallel_forward, qkv_native_lora_apply, fixed_mlp_forward, fixed_head,
)


@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('defer_bias', [False, True])
@pytest.mark.parametrize('return_bias', [False, True])
def test_parallel_projection_preserves_bias_and_return_contract(split, defer_bias, return_bias):
    torch.manual_seed(9)
    x = torch.randn(2, 5, 4, dtype=torch.float64)
    w = torch.randn(9, 4, dtype=torch.float64)
    b = torch.randn(9, dtype=torch.float64)
    module = SimpleNamespace(weight=w, bias=b, skip_bias_add=defer_bias,
                             return_bias=return_bias, _jev_projection_splits=(3, 3, 3) if split else None)
    result = parallel_forward(module, x)
    expected = torch.nn.functional.linear(x, w, None if defer_bias else b)
    if return_bias:
        result, bias = result
        assert bias is (b if defer_bias else None)
    torch.testing.assert_close(result, expected)


def test_qkv_retains_native_adapter_application_and_unequal_slices():
    torch.manual_seed(19)
    x = torch.randn(5, 4, dtype=torch.float64)
    w = torch.randn(8, 4, dtype=torch.float64)
    b = torch.randn(8, dtype=torch.float64)
    calls = []

    def native(inputs, base):
        assert inputs is x
        calls.append(base)
        return base + 7

    module = SimpleNamespace(output_slices=(4, 2, 2), base_layer=SimpleNamespace(weight=w),
                             _apply_lora_to_output=native)
    result = qkv_native_lora_apply(module, x, b)
    assert len(calls) == 1
    torch.testing.assert_close(result, torch.nn.functional.linear(x, w, b) + 7)


def test_mlp_and_head_use_actual_weights():
    torch.manual_seed(29)
    x = torch.randn(3, 4, dtype=torch.float64)
    w = torch.randn(12, 4, dtype=torch.float64)
    down = torch.randn(4, 6, dtype=torch.float64)
    module = SimpleNamespace(gate_up_proj=SimpleNamespace(weight=w),
                             down_proj=lambda t: (t @ down.T, None))
    gate, up = w.chunk(2, 0)
    expected = (torch.nn.functional.silu(x @ gate.T) * (x @ up.T)) @ down.T
    torch.testing.assert_close(fixed_mlp_forward(module, x), expected)
    head = SimpleNamespace(weight=w)
    processor = SimpleNamespace(head_dtype=x.dtype)
    torch.testing.assert_close(fixed_head(processor, head, x, None), x @ w.T)
    processor.head_dtype = torch.float32
    with pytest.raises(ValueError, match='model dtype'):
        fixed_head(processor, head, x, None)

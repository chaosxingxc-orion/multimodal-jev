import pytest
import torch

from tools.reports.context_attention_kernel_probe import layouts, summarize, MODES


def test_strided_value_has_native_fused_stride_without_changing_elements():
    q,k,v=[torch.randn(7,h,4) for h in (4,2,2)]
    a,b,c=layouts(q,k,v,'strided_v')
    assert torch.equal(q,a) and torch.equal(k,b) and torch.equal(v,c)
    assert c.stride()==(32,4,1) and not c.is_contiguous()
    with pytest.raises(ValueError):layouts(q,k,v,'unknown')


def test_comparison_separates_shape_invariance_from_original_numeric_agreement():
    a=torch.zeros(4,2,2);b=a+1
    payload=dict(rows=[dict(expected=a)],recorded_batch=b)
    outputs={f'p{p}_{l}_b{size}':b.clone() for p,l in MODES for size in (1,8,32)}
    result=summarize(payload,outputs)
    entry=result['p2_contiguous']['32']
    assert entry['vs_same_mode_single']['exact']
    assert not entry['vs_original_hf']['exact']

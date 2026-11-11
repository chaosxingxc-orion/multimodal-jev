from itertools import product
import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.span_parity import compare_span_policies


def test_joint_gate_catches_accumulated_error_that_per_span_gate_accepts():
    b=[torch.tensor([.5,.5]).log() for _ in range(3)]
    a=[torch.tensor([.505,.495]).log() for _ in range(3)]
    r=compare_span_policies(a,b)
    assert r['per_span_passed'] and not r['joint_ratio_passed'] and not r['passed']
    ratios=[torch.exp(sum(a[j][i]-b[j][i] for j,i in enumerate(choice))).item()
            for choice in product(range(2),repeat=3)]
    assert r['joint_ratio_min']==pytest.approx(min(ratios),abs=1e-7)
    assert r['joint_ratio_max']==pytest.approx(max(ratios),abs=1e-7)


def test_identical_policies_pass_and_bad_normalization_rejected():
    p=[torch.tensor([.1,.3,.6]).log(),torch.tensor([.2,.8]).log()]
    r=compare_span_policies(p,p)
    assert r['passed'] and r['joint_ratio_min']==r['joint_ratio_max']==1
    with pytest.raises(ValueError,match='normalized'):compare_span_policies([p[0]+1],[p[0]])

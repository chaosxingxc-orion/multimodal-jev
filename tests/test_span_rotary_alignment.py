from types import SimpleNamespace
import pytest
torch = pytest.importorskip('torch')
from multimodal_jev.span_rotary_alignment import hf_order_rotary


def test_rotary_identity_at_zero_and_independent_pair_rotation():
    torch.manual_seed(4)
    q = torch.randn(5, 24)
    k = torch.randn(5, 8)
    module = SimpleNamespace(is_neox_style=True, rotary_dim=8, head_size=8, base=10000.)
    a, b = hf_order_rotary(module, torch.arange(5), q, k)
    assert torch.equal(a[0], q[0]) and torch.equal(b[0], k[0])
    for original, actual in ((q, a), (k, b)):
        expected = original.reshape(5, -1, 8).clone()
        for t in range(5):
            for j in range(4):
                angle = torch.tensor(t / (10000. ** (2*j/8)))
                x, y = original.reshape(5, -1, 8)[t, :, j], original.reshape(5, -1, 8)[t, :, j+4]
                expected[t, :, j] = x*angle.cos()-y*angle.sin()
                expected[t, :, j+4] = y*angle.cos()+x*angle.sin()
        torch.testing.assert_close(actual, expected.reshape_as(actual))
    module.is_neox_style = False
    with pytest.raises(ValueError, match='NeoX'):
        hf_order_rotary(module, torch.arange(5), q, k)

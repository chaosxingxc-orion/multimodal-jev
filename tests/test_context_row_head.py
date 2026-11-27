import copy
import torch
import pytest
from multimodal_jev.context_decision_head import ContextDecisionHead
from multimodal_jev.context_row_head import install_row_head


def test_row_head_matches_unpadded_reference_and_preserves_gradients():
    torch.manual_seed(317)
    h=ContextDecisionHead(8,2,16).double()
    ref=copy.deepcopy(h)
    install_row_head(h)
    x=torch.randn(2,9,8,dtype=torch.float64,requires_grad=True)
    y=x.detach().clone().requires_grad_()
    valid=torch.arange(9)[None]<torch.tensor([[6],[9]])
    pos=torch.tensor([[2,5],[4,8]])
    a=h.score_logits(x,pos,valid)
    b=torch.cat([ref.score_logits(y[i:i+1,:n],pos[i:i+1],valid[i:i+1,:n]) for i,n in enumerate([6,9])])
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    a.square().sum().backward();b.square().sum().backward()
    torch.testing.assert_close(x.grad,y.grad,atol=0,rtol=0)
    for p,q in zip(h.parameters(),ref.parameters(),strict=True):
        torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
    with pytest.raises(ValueError):install_row_head(h)
    valid[0,0]=False
    with pytest.raises(ValueError,match='right padding'):h.score_logits(x,pos,valid)


@torch.inference_mode()
def test_eval_validation_once_keeps_singleton_math_and_rejects_invalid_padding():
    head = ContextDecisionHead(8, 2, 16).eval()
    baseline = copy.deepcopy(head)
    install_row_head(baseline)
    install_row_head(head, single_validation=True)
    hidden = torch.randn(3, 12, 8)
    valid = torch.arange(12)[None] < torch.tensor([[8], [12], [10]])
    positions = torch.tensor([[7], [3], [9]])
    torch.testing.assert_close(head.score_logits(hidden, positions, valid),
                               baseline.score_logits(hidden, positions, valid), atol=0, rtol=0)
    positions[0, 0] = 9
    with pytest.raises(ValueError, match='marker in padding'):
        head.score_logits(hidden, positions, valid)
    positions[0, 0] = 7
    hidden[0, 11, 0] = torch.nan
    with pytest.raises(ValueError, match='finite hidden states'):
        head.score_logits(hidden, positions, valid)

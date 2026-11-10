import sys
from types import SimpleNamespace
import pytest

torch=pytest.importorskip('torch')
from multimodal_jev.verl_cpu_logprobs import cpu_logprobs,native_actor_cpu_logprobs


def test_cpu_forward_and_gradient_match_cross_entropy_on_noncontiguous_logits():
    torch.manual_seed(17)
    logits=torch.randn(3,7,11,dtype=torch.float64,requires_grad=True)
    labels=torch.randint(11,(3,4))
    selected=logits[:,:4]
    assert not selected.is_contiguous()
    result=cpu_logprobs(selected,labels)
    expected=-torch.nn.functional.cross_entropy(selected.reshape(-1,11),labels.reshape(-1),reduction='none').reshape(3,4)
    torch.testing.assert_close(result,expected,atol=1e-12,rtol=1e-12)
    weight=torch.randn(3,4,dtype=torch.float64)
    left=torch.autograd.grad((result*weight).sum(),logits,retain_graph=True)[0]
    right=torch.autograd.grad((expected*weight).sum(),logits)[0]
    torch.testing.assert_close(left,right,atol=1e-12,rtol=1e-12)
    assert not left[:,4:].any()


def test_scoped_dispatch_restored_after_failure(monkeypatch):
    original=object();actor=SimpleNamespace(logprobs_from_logits=original)
    monkeypatch.setitem(sys.modules,'verl.workers.actor',SimpleNamespace(dp_actor=actor))
    with pytest.raises(RuntimeError):
        with native_actor_cpu_logprobs():
            assert actor.logprobs_from_logits is cpu_logprobs
            raise RuntimeError('fixture error')
    assert actor.logprobs_from_logits is original


def test_invalid_label_shape_rejected():
    with pytest.raises(ValueError,match='aligned'):
        cpu_logprobs(torch.zeros(2,3,4),torch.zeros(2,2,dtype=torch.long))

import sys
from types import SimpleNamespace
import pytest

torch=pytest.importorskip('torch')
from multimodal_jev.verl_cpu_logprobs import cpu_logprobs,native_actor_cpu_logprobs,native_actor_cpu_device


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


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_preserves_small_negative_logprob_and_gradient(dtype):
    # A confident EOS-like prediction exposes loss of normalization precision.
    logits=torch.tensor([[0.,-12.,-13.],[-3.,0.,-2.]],dtype=dtype,requires_grad=True)
    original=logits.detach().clone()
    labels=torch.tensor([0,2])
    with torch.autocast('cpu',dtype=torch.bfloat16):
        result=cpu_logprobs(logits,labels)
    expected=-torch.nn.functional.cross_entropy(logits.float(),labels,reduction='none')
    assert result.dtype==torch.float32
    assert -1e-4 < result[0].item() < 0.
    torch.testing.assert_close(result,expected,rtol=1e-6,atol=1e-9)
    actual_grad=torch.autograd.grad(result.sum(),logits,retain_graph=True)[0]
    expected_grad=torch.autograd.grad(expected.sum(),logits)[0]
    torch.testing.assert_close(actual_grad,expected_grad,rtol=0,atol=0)
    assert actual_grad[0,0] > 0
    torch.testing.assert_close(logits.detach(),original,rtol=0,atol=0)


def test_cpu_device_override_restores_on_exception(monkeypatch):
    original = object()
    module = SimpleNamespace(get_device_id=original)
    monkeypatch.setitem(sys.modules, 'verl.workers.actor', SimpleNamespace(dp_actor=module))
    actor = SimpleNamespace(device_name='cpu', actor_module=torch.nn.Linear(2, 3))
    with pytest.raises(RuntimeError):
        with native_actor_cpu_device(actor):
            assert module.get_device_id() == torch.device('cpu')
            raise RuntimeError('fixture')
    assert module.get_device_id is original


def test_device_override_rejects_non_cpu_actor_before_mutation(monkeypatch):
    original = object()
    module = SimpleNamespace(get_device_id=original)
    monkeypatch.setitem(sys.modules, 'verl.workers.actor', SimpleNamespace(dp_actor=module))
    actor = SimpleNamespace(device_name='cpu', actor_module=torch.nn.Linear(2, 3, device='meta'))
    with pytest.raises(ValueError, match='CPU actor'):
        with native_actor_cpu_device(actor):
            pass
    assert module.get_device_id is original

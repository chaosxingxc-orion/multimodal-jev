import sys
from types import SimpleNamespace

import pytest
import torch

from multimodal_jev import omni_logprob_observer as observer


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('batch', [False, True])
def test_chunked_reference_matches_cross_entropy_without_modifying_inputs(dtype, batch):
    torch.manual_seed(17)
    logits = torch.randn(2, 9, 13).to(dtype)[:, 1:8]
    labels = torch.randint(13, (2, 7))
    if not batch:
        logits, labels = logits[0], labels[0]
    before = logits.clone()
    actual = observer.reference_logprobs(logits, labels, rows_per_chunk=2)
    expected = -torch.nn.functional.cross_entropy(logits.float().reshape(-1,13),
        labels.flatten(), reduction='none').reshape(labels.shape)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert torch.equal(logits, before)


def test_returns_original_output_and_records_real_discrepancy_once():
    logits = torch.tensor([[0., -12., -13.], [-3., 0., -2.]], dtype=torch.bfloat16)
    labels = torch.tensor([0, 2]); calls=[]; records=[]
    native = logits.log_softmax(-1).gather(-1,labels[:,None]).squeeze(-1)
    def original(*args, **kwargs):
        calls.append(kwargs);return native
    with torch.no_grad():
        output = observer.observed_call(original,logits,labels,records,expected_device='cpu',inplace_backward=False)
    assert output is native and calls == [dict(inplace_backward=False)]
    assert records[0]['max_abs_error'] > 0
    assert records[0]['native_log_probs'][0] == 0.
    assert records[0]['reference_log_probs'][0] < 0.
    assert records[0]['label_ids'] == labels.tolist()


def test_fp32_reference_overflow_is_explicit_failure():
    logits=torch.tensor([[3e38,-3e38]],dtype=torch.float32)
    with pytest.raises(ValueError,match='nonfinite FP32'):
        observer.reference_logprobs(logits,torch.tensor([1]))


def test_failure_not_retried_or_replaced():
    calls=[];records=[]
    def original(*args, **kwargs):
        calls.append(1);raise RuntimeError('native failed')
    with torch.no_grad(), pytest.raises(RuntimeError,match='native failed'):
        observer.observed_call(original,torch.zeros(2,3),torch.zeros(2,dtype=torch.long),records,expected_device='cpu')
    assert calls == [1] and records == []


@pytest.mark.parametrize('kind', ['gradient','device','nan_input','bad_label','nan_output','positive','shape'])
def test_invalid_inputs_or_outputs_rejected(kind):
    logits=torch.zeros(2,3);labels=torch.zeros(2,dtype=torch.long);calls=[]
    if kind=='nan_input':logits[0,0]=float('nan')
    if kind=='bad_label':labels[0]=3
    output=torch.full((2,),-1.)
    if kind=='nan_output':output[0]=float('nan')
    if kind=='positive':output[0]=.1
    if kind=='shape':output=output[:1]
    def original(*args,**kwargs):calls.append(1);return output
    with torch.set_grad_enabled(kind=='gradient'),pytest.raises(ValueError):
        observer.observed_call(original,logits,labels,[],expected_device='npu' if kind=='device' else 'cpu')
    assert len(calls)==(1 if kind in ('nan_output','positive','shape') else 0)


def modules(monkeypatch):
    def original(*args,**kwargs):raise AssertionError('not called')
    actor=SimpleNamespace(logprobs_from_logits=original)
    functional=SimpleNamespace(logprobs_from_logits=original,
        FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE=False,NPU_CROSS_ENTROPY_LOSS_AVAILABLE=True)
    monkeypatch.setitem(sys.modules,'verl.workers.actor',SimpleNamespace(dp_actor=actor))
    monkeypatch.setitem(sys.modules,'verl.utils',SimpleNamespace(torch_functional=functional))
    monkeypatch.setattr(observer,'file_hash',lambda path:observer.PINNED_SOURCE)
    return actor,functional,original


def test_context_restores_after_error_and_rejects_nesting(monkeypatch):
    actor,_,original=modules(monkeypatch)
    with pytest.raises(RuntimeError):
        with observer.observe_native_logprobs() as report:
            assert actor.logprobs_from_logits is not original and report['npu_available']
            with pytest.raises(ValueError):
                with observer.observe_native_logprobs():pass
            raise RuntimeError('fixture')
    assert actor.logprobs_from_logits is original


@pytest.mark.parametrize('kind',['source','replacement','flash','no_npu','chunk'])
def test_context_rejects_wrong_dispatch_before_mutation(monkeypatch,kind):
    actor,functional,original=modules(monkeypatch)
    if kind=='source':monkeypatch.setattr(observer,'file_hash',lambda path:'b'*64)
    if kind=='replacement':actor.logprobs_from_logits=lambda *a:None
    if kind=='flash':functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE=True
    if kind=='no_npu':functional.NPU_CROSS_ENTROPY_LOSS_AVAILABLE=False
    before=actor.logprobs_from_logits
    with pytest.raises(ValueError):
        with observer.observe_native_logprobs(rows_per_chunk=0 if kind=='chunk' else 16):pass
    assert actor.logprobs_from_logits is before

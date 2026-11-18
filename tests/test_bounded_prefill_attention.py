import pytest
import torch
from multimodal_jev.bounded_prefill_attention import bounded_prefill_kernel


def test_groups_preserve_boundaries_values_and_gradients():
    lengths=[1,3,2,4,1,2,3,1,2,4]
    ends=torch.tensor([0,*lengths],dtype=torch.int32).cumsum(0,dtype=torch.int32)
    q=torch.randn(sum(lengths),4,2,requires_grad=True)
    k=torch.randn(sum(lengths),2,2,requires_grad=True);v=torch.randn_like(k,requires_grad=True)
    calls=[]
    def oracle(a,b,c,cq,ck,**kw):
        calls.append(cq.tolist())
        assert torch.equal(cq,ck) and int(cq[-1])==len(a)
        assert kw['max_seqlen_q']==int((cq[1:]-cq[:-1]).max())
        assert a.is_contiguous() and b.is_contiguous() and c.is_contiguous()
        return a+b.repeat_interleave(2,1)+c.repeat_interleave(2,1)
    out=bounded_prefill_kernel(q,k,v,ends,ends,kernel=oracle,
        causal=True,dropout_p=0.,softmax_scale=1.,max_seqlen_q=4,max_seqlen_k=4)
    expected=q+k.repeat_interleave(2,1)+v.repeat_interleave(2,1)
    assert torch.equal(out,expected) and [len(c)-1 for c in calls]==[8,2]
    out.sum().backward()
    assert torch.equal(q.grad,torch.ones_like(q))
    assert torch.equal(k.grad,torch.full_like(k,2))


def test_partial_or_noncausal_inputs_fail_closed():
    q=torch.ones(3,2,4);k=torch.ones(3,1,4);ends=torch.tensor([0,1,3],dtype=torch.int32)
    for bad in (torch.tensor([0,1,2],dtype=torch.int32),torch.tensor([0,0,3],dtype=torch.int32)):
        with pytest.raises(ValueError):bounded_prefill_kernel(q,k,k,bad,bad,causal=True,dropout_p=0.)
    with pytest.raises(ValueError):bounded_prefill_kernel(q,k,k,ends,ends,causal=False,dropout_p=0.)


def test_probe_binds_only_new_worker_and_keeps_hf_reference():
    from tools.reports.context_bounded_attention_probe import namespace
    constants=str(namespace()['phase'].__code__.co_consts)
    assert 'context_bounded_attention_worker' in constants
    assert 'tools.reports.context_language_stage_capture' in namespace()['phase'].__code__.co_names


def test_causal_attention_and_gradients_match_independent_requests():
    # Real CPU attention oracle: tests causal boundaries and backward, not an
    # Ascend arithmetic claim. Include a partial final group and uneven lengths.
    torch.manual_seed(17)
    lengths = [2, 5, 1, 3, 4, 2, 6, 1, 3]
    ends = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    tensors = [torch.randn(sum(lengths), h, 4, dtype=torch.float64, requires_grad=True)
               for h in (4, 2, 2)]
    references = [t.detach().clone().requires_grad_() for t in tensors]

    def attention(q, k, v):
        k, v = [t.repeat_interleave(2, dim=1) for t in (k, v)]
        logits = torch.einsum('thd,shd->hts', q, k) * .5
        mask = torch.ones(len(q), len(q), dtype=torch.bool).triu(1)
        return torch.einsum('hts,shd->thd', logits.masked_fill(mask, -float('inf')).softmax(-1), v)

    def oracle(q, k, v, cq, ck, **kwargs):
        return torch.cat([attention(q[a:b], k[a:b], v[a:b])
                          for a, b in zip(cq.tolist()[:-1], cq.tolist()[1:])])

    actual = bounded_prefill_kernel(*tensors, ends, ends, kernel=oracle, causal=True,
                                    dropout_p=0., softmax_scale=.5)
    expected = torch.cat([attention(*(t[a:b] for t in references))
                          for a, b in zip(ends.tolist()[:-1], ends.tolist()[1:])])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward(); (expected * weights).sum().backward()
    for actual_tensor, reference_tensor in zip(tensors, references):
        torch.testing.assert_close(actual_tensor.grad, reference_tensor.grad, rtol=0, atol=0)

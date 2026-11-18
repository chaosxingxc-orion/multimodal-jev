"""Requires the fixed vLLM image; local absence is an explicit skip."""
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('vllm')
from multimodal_jev.context_decision_head import ContextDecisionHead
from multimodal_jev.vllm_context_pooler import ContextProbabilityPooler


class Metadata:
    def __init__(self, prompts, extras, partial=False, finished=True, lengths=None):
        self.prompts = prompts
        self.pooling_params = [SimpleNamespace(extra_kwargs=x) for x in extras]
        lens = [len(x) for x in prompts] if lengths is None else lengths
        self.cursor = SimpleNamespace(
            is_partial_prefill=lambda: partial,
            get_finished_mask=lambda: [finished]*len(prompts),
            num_scheduled_tokens_cpu=torch.tensor(lens))

    def get_pooling_cursor(self): return self.cursor
    def get_prompt_token_ids_cpu(self): return self.prompts


def test_real_pooler_interface_and_explicit_profiling_budget():
    torch.manual_seed(317)
    head = ContextDecisionHead(12, 3, 24).eval()
    pooler = ContextProbabilityPooler(head, 99, 'a'*64, profiling_decisions=3)
    assert pooler.get_supported_tasks() == {'token_classify'}
    assert pooler.get_pooling_updates('token_classify').requires_token_ids
    h = torch.randn(7, 12)
    p = torch.tensor([1, 2, 99, 3, 99, 4, 99])
    extra = dict(policy_id='a'*64, marker_positions=[2, 4, 6], expected_input_ids=p.tolist())
    got = pooler(h, Metadata([p], [extra]))[0]
    expected = head(h[None], torch.tensor([[2, 4, 6]]), torch.ones(1, 7, dtype=torch.bool))[0, :, None]
    torch.testing.assert_close(got, expected, atol=0, rtol=0)
    for partial, finished in [(True, True), (False, False)]:
        with pytest.raises(ValueError): pooler(h, Metadata([p], [extra], partial, finished))
    dummy = Metadata([torch.zeros(7, dtype=torch.long)], [None])
    with pytest.raises(ValueError): pooler(h, dummy)
    pooler.memory_profiling = True
    assert pooler(h, dummy)[0].shape == (3, 1)
    assert (dummy.prompts[0] == 0).all()
    with pytest.raises(ValueError): pooler(h, Metadata([p], [extra]))
    pooler.memory_profiling = False
    with pytest.raises(ValueError): pooler(h, dummy)


def test_dummy_pooler_run_rectangular_rows_use_scheduled_lengths():
    """vLLM profiles with one rectangular int32 block: the last scheduled
    request carries the token remainder, so its row is shorter than its
    scheduled length. The pooler must rebuild exact-length synthetic zeros."""
    torch.manual_seed(317)
    head = ContextDecisionHead(12, 3, 24).eval()
    pooler = ContextProbabilityPooler(head, 99, 'a'*64, profiling_decisions=3)
    rows = [torch.zeros(2, dtype=torch.int32) for _ in range(3)]
    metadata = Metadata(rows, [None]*3, lengths=[2, 2, 3])
    pooler.memory_profiling = True
    outputs = pooler(torch.randn(7, 12), metadata)
    assert [tuple(o.shape) for o in outputs] == [(2, 1), (2, 1), (3, 1)]
    assert all((row == 0).all() for row in rows)


def test_native_int32_metadata_startup_and_real_request():
    from vllm import PoolingParams
    from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates, PoolingCursor

    torch.manual_seed(317)
    head = ContextDecisionHead(12, 3, 24).eval()
    pooler = ContextProbabilityPooler(head, 99, 'a'*64, profiling_decisions=3)
    h = torch.randn(7, 12)
    ids = torch.zeros((1, 7), dtype=torch.int32)
    lens = torch.tensor([7])
    cursor = PoolingCursor(torch.tensor([0]), torch.tensor([6]), lens, lens,
                           lens, False, [True])
    params = PoolingParams(task='token_classify')
    metadata = PoolingMetadata(lens, ids, ids, [params], [PoolingStates()], cursor)
    pooler.memory_profiling = True
    got = pooler(h, metadata)[0]
    expected = head(h[None], torch.tensor([[4, 5, 6]]),
                    torch.ones(1, 7, dtype=torch.bool))[0, :, None]
    torch.testing.assert_close(got, expected, atol=0, rtol=0)
    assert ids.dtype == torch.int32 and (ids == 0).all()
    pooler.memory_profiling = False
    ids[0] = torch.tensor([1, 2, 99, 3, 99, 4, 99])
    params.extra_kwargs = dict(policy_id='a'*64, marker_positions=[2, 4, 6],
                               expected_input_ids=ids[0].tolist())
    got = pooler(h, metadata)[0]
    expected = head(h[None], torch.tensor([[2, 4, 6]]),
                    torch.ones(1, 7, dtype=torch.bool))[0, :, None]
    torch.testing.assert_close(got, expected, atol=0, rtol=0)
    for dtype in (torch.float32, torch.bool):
        metadata.prompt_token_ids_cpu = ids.to(dtype)
        with pytest.raises(ValueError, match='int32 or int64'):
            pooler(h, metadata)

import asyncio
import copy
import sys
from types import SimpleNamespace as NS
import pytest

from multimodal_jev.async_audio_rollout import AsyncAudioRollout


def fixture(monkeypatch, mode='success'):
    import multimodal_jev.async_audio_rollout as module
    import multimodal_jev.vllm_audio_features as features
    monkeypatch.setitem(sys.modules, 'vllm', NS(SamplingParams=lambda **kwargs: kwargs))
    monkeypatch.setattr(features, 'enable_feature_passthrough', lambda: None)
    monkeypatch.setattr(module, 'hf_prompt', lambda *args: {'input_ids': [NS(tolist=lambda: [1, 2, 3])]})
    monkeypatch.setattr(module, 'vllm_request', lambda *args: {'prompt': 'raw<|audio_pad|>', 'multi_modal_data': {'audio': 'features'}})
    class Engine:
        def __init__(self): self.calls, self.aborts = [], []
        async def generate(self, **kwargs):
            self.calls.append(copy.deepcopy(kwargs))
            if mode == 'error': raise RuntimeError('engine failure')
            if mode == 'cancel': raise asyncio.CancelledError()
            if mode == 'empty': return
            request_id = kwargs['request_id'] if mode != 'identity' else 'wrong-id'
            yield NS(request_id=request_id, finished=False)
            yield NS(request_id=request_id, finished=mode != 'unfinished',
                     prompt_token_ids=[1, 2, 4] if mode == 'prompt' else [1, 2, 3],
                     outputs=[NS(index=i, token_ids=[4, 5], logprobs=[{4:NS(logprob=-.2)}, {5:NS(logprob=-.3)}],
                                 finish_reason='stop') for i in ([0] if mode == 'partial_group' else [1, 0])])
        async def abort(self, request_id): self.aborts.append(request_id)
    engine = Engine()
    runner = AsyncAudioRollout(engine, NS(tokenizer=NS(decode=lambda *a, **k: 'decoded')), 'policy', 32)
    config = {'temperature': 1., 'top_p': 1., 'top_k': -1, 'n': 2, 'max_tokens': 8, 'seed': 17}
    return runner, engine, config


def test_completed_groups_preserve_identity_and_sampling_config(monkeypatch):
    runner, engine, config = fixture(monkeypatch)
    before = copy.deepcopy(config)
    rows = asyncio.run(runner.sample([NS(example_id='a'), NS(example_id='b')], [0, 1], config))
    assert config == before and not engine.aborts
    assert [(r.example_id, r.replicate) for r in rows] == [('a',0),('a',1),('b',0),('b',1)]
    assert len(set(runner.last_request_ids)) == 2
    assert runner.last_generation_budget['generated_tokens'] == 8
    assert all(c['prompt']['prompt'] == 'raw<|audio_pad|>' for c in engine.calls)
    assert all('prompt_token_ids' not in c['prompt'] for c in engine.calls)


@pytest.mark.parametrize('mode', ['error', 'cancel', 'empty', 'identity', 'unfinished', 'prompt', 'partial_group'])
def test_failed_or_cancelled_requests_never_publish_complete_budget(monkeypatch, mode):
    runner, engine, config = fixture(monkeypatch, mode)
    runner.last_generation_budget = {'stale': True}
    with pytest.raises((ValueError, RuntimeError, asyncio.CancelledError)):
        asyncio.run(runner.sample([NS(example_id='a')], [0], config))
    assert runner.last_generation_budget is None and runner.last_request_ids is None
    assert len(engine.aborts) == (0 if mode in ['prompt', 'partial_group'] else 1)


def test_context_admission_precedes_engine_call(monkeypatch):
    runner, engine, config = fixture(monkeypatch)
    runner.max_model_len = 10
    with pytest.raises(ValueError, match='context'):
        asyncio.run(runner.sample([NS(example_id='a')], [0], config))
    assert not engine.calls

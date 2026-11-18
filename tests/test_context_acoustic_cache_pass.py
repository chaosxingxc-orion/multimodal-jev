from types import SimpleNamespace

import pytest
import torch

from tools.reports.measure_context_acoustic_candidate import fresh_audio_pass


def test_repeated_inputs_force_encoder_cache_misses_without_changing_tensors():
    calls = []; cache = set(); misses = []
    def encode(prompts, **kwargs):
        calls.append((prompts, kwargs))
        for p in prompts:
            key = p['multi_modal_uuids']['audio'][0]
            if key not in cache: misses.append(key); cache.add(key)
        return 'same-output'
    engine = SimpleNamespace(encode=encode); owner = SimpleNamespace(engine=engine)
    audio = torch.randn(2, 100)
    prompt = dict(prompt_token_ids=[1, 2], multi_modal_data={'audio': audio},
                  multi_modal_uuids={'audio': ['stable-request']})
    for index in range(4):
        with fresh_audio_pass(owner, index):
            assert owner.engine.encode([prompt], policy_id='fixed') == 'same-output'
        assert owner.engine is engine
    assert len(set(misses)) == 4
    assert prompt['multi_modal_uuids']['audio'] == ['stable-request']
    for prompts, kwargs in calls:
        assert prompts[0]['multi_modal_data']['audio'] is audio
        assert prompts[0]['prompt_token_ids'] is prompt['prompt_token_ids']
        assert kwargs == {'policy_id': 'fixed'}


def test_engine_restored_on_failure():
    engine = object(); owner = SimpleNamespace(engine=engine)
    with pytest.raises(RuntimeError):
        with fresh_audio_pass(owner, 1): raise RuntimeError('failure')
    assert owner.engine is engine

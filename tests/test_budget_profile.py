import pytest
from multimodal_jev.budget_profile import schedule_profile


def test_gold_and_capped_sample_exposures_with_repeated_schedule():
    rows = {'a': {'prompt_tokens': 10, 'gold_tokens': 2, 'audio_frames': 50},
            'b': {'prompt_tokens': 20, 'gold_tokens': 4, 'audio_frames': 100}}
    batches = [['a', 'b'], ['a']]
    sft = schedule_profile(rows, batches)
    assert sft['total']['gradient_tokens'] == 48
    assert sft['total']['audio_frames'] == 200
    assert sft['total']['no_grad_tokens'] == 0
    rl = schedule_profile(rows, batches, samples=4, max_response_tokens=32, no_grad_passes=1)
    assert rl['total']['gradient_tokens'] == 544
    assert rl['total']['no_grad_tokens'] == 544
    assert rl['total']['response_tokens'] == 384
    assert rl['total']['audio_frames'] == 800
    assert rl['total']['optimizer_steps'] == 2
    assert rl['largest_batch_by_field']['gradient_tokens'] == 376


@pytest.mark.parametrize('kwargs', [{'samples': 0}, {'samples': True},
                                  {'max_response_tokens': 0}])
def test_invalid_profile_parameters(kwargs):
    with pytest.raises(ValueError):
        schedule_profile({}, [['a']], **kwargs)


def test_duplicate_within_batch_rejected():
    with pytest.raises(ValueError):
        schedule_profile({}, [['a', 'a']])

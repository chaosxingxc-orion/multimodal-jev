import json
from pathlib import Path
import pytest

from multimodal_jev.budget_control import (
    BudgetController, contract_hashes, load_budget_controller, validate_training_reservation)
from multimodal_jev.training_budget import batch_budget

ROWS = {'a': {'prompt_tokens': 10, 'gold_tokens': 2, 'audio_frames': 50},
        'b': {'prompt_tokens': 20, 'gold_tokens': 4, 'audio_frames': 100}}


def controller(**kwargs):
    return BudgetController({'training_tokens': 200, 'generated_tokens': 100}, ROWS,
                            samples=2, response_cap=8, no_grad_passes=1, **kwargs)


def test_rl_noop_counts_parity_and_releases_unused_gradient_reservation():
    budget = controller()
    ids = ['a', 'b']
    assert budget.reservation(ids)[0] == {'training_tokens': 184, 'generated_tokens': 32}
    actual = batch_budget([10, 10, 20, 20], [2, 3, 4, 5], [50, 50, 100, 100],
                          gradient_passes=0, no_grad_passes=1, optimizer_steps=0)
    generation = {'requests': 2, 'responses': 4, 'generated_tokens': 14, 'replicated_prompt_tokens': 60}
    budget.consume(ids, actual, generation)
    assert budget.used == {'training_tokens': 74, 'generated_tokens': 14}
    assert not budget.admit(ids)
    assert budget.report(ids)['next_reservation']['training_tokens'] == 184
    # A resumed controller must recover identical state from actual receipts.
    resumed = controller();resumed.consume(ids, actual, generation)
    assert resumed.report(ids) == budget.report(ids)


def test_exact_sft_boundary_and_first_batch_rejection():
    budget = BudgetController({'training_tokens': 36, 'generated_tokens': 0}, ROWS)
    actual = batch_budget([10, 20], [2, 4], [50, 100],
                          gradient_passes=1, no_grad_passes=0, optimizer_steps=1)
    assert budget.admit(['a', 'b'])
    budget.consume(['a', 'b'], actual)
    assert budget.used['training_tokens'] == 36
    assert not budget.admit(['a'])
    empty = BudgetController({'training_tokens': 0, 'generated_tokens': 0}, ROWS)
    assert not empty.admit(['a'])
    with pytest.raises(ValueError, match='exceeds reserved'):
        empty.consume(['a', 'b'], actual)


def test_generation_limit_and_missing_accounting_fail_closed():
    budget = BudgetController({'training_tokens': 1000, 'generated_tokens': 31}, ROWS,
                              samples=2, response_cap=8, no_grad_passes=1)
    assert not budget.admit(['a', 'b'])
    budget = controller()
    actual = budget.reservation(['a', 'b'])[1]
    with pytest.raises(ValueError, match='generation accounting'):
        budget.consume(['a', 'b'], actual)
    with pytest.raises(ValueError, match='fields'):
        budget.consume(['a'], {})
    with pytest.raises(ValueError, match='counts differ'):
        budget.consume(['a', 'b'], actual, {'requests': 2, 'responses': 4,
                                           'generated_tokens': 100, 'replicated_prompt_tokens': 60})
    assert budget.used == {'training_tokens': 0, 'generated_tokens': 0}


def test_actual_tensors_must_fit_reservation_before_forward():
    upper = controller().reservation(['a'])[1]
    actual = batch_budget([10, 10], [2, 3], [50, 50],
                          gradient_passes=1, no_grad_passes=1, optimizer_steps=1)
    validate_training_reservation(actual, upper)
    with pytest.raises(ValueError, match='processor lengths'):
        validate_training_reservation({**actual, 'prompt_tokens': 21}, upper)
    with pytest.raises(ValueError, match='response lengths'):
        validate_training_reservation({**actual, 'response_tokens': 17}, upper)


@pytest.mark.parametrize('limits', [{'training_tokens': True, 'generated_tokens': 0},
                                    {'training_tokens': -1, 'generated_tokens': 0},
                                    {'training_tokens': 100}])
def test_invalid_limits(limits):
    with pytest.raises(ValueError, match='limits'):
        BudgetController(limits, ROWS)


def profiled_fixture(tmp_path):
    from test_rl_preflight import fixture
    from multimodal_jev.rl_preflight import validate_schedule
    from multimodal_jev.baseline import file_hash
    cfg = fixture(tmp_path)
    cfg['audio_input_mode'] = 'actor_features';cfg['engine']['max_model_len'] = 128
    profile = {'contract_sha256': contract_hashes(), 'preflight': validate_schedule(cfg),
               'examples': {'ex': ROWS['a']}}
    path = tmp_path/'profile.json';path.write_text(json.dumps(profile))
    cfg['budget_control'] = {'profile_path': str(path), 'profile_sha256': file_hash(path),
                             'limits': {'training_tokens': 104, 'generated_tokens': 32}}
    return cfg


def test_profile_identity_and_contract_binding(tmp_path):
    from multimodal_jev.baseline import file_hash
    cfg = profiled_fixture(tmp_path)
    assert load_budget_controller(cfg).admit(['ex'])
    path = Path(cfg['budget_control']['profile_path'])
    d = json.loads(path.read_text());d['contract_sha256'] = {};path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='hash changed'):load_budget_controller(cfg)
    cfg['budget_control']['profile_sha256'] = file_hash(path)
    with pytest.raises(ValueError, match='contract changed'):load_budget_controller(cfg)


def test_changed_audio_rejected(tmp_path):
    cfg = profiled_fixture(tmp_path)
    (tmp_path/'audio.wav').write_bytes(b'changed')
    with pytest.raises(ValueError, match='audio changed'):load_budget_controller(cfg)

"""Contract tests for the annexed curriculum candidate; no training dispatch."""
import json
from pathlib import Path

import pytest
import torch

from multimodal_jev.context_curriculum_credit import (
    plackett_luce_log_prob, schedule,
)
from multimodal_jev.context_dual_credit import prepare
from tools.reports import context_curriculum_candidate as candidate

ROOT = Path(__file__).resolve().parents[1]
DESIGN = json.loads((ROOT / 'configs/experiments/context-annealed-curriculum-v1.json').read_text())


def runtime_config():
    config = dict(DESIGN)
    config.update(
        experiment_kind='formal_source', start_update=0, stop_update=212,
        stage='task', resume_root=None, engineering_prefix=False,
        training_dispatch_allowed=True,
        evaluation_steps=DESIGN['evaluation']['evaluation_steps'],
        dual_arm='joint', seed=317,
        batch_config=DESIGN['topology']['batch_config'])
    return config


def make_record(logits, labels, *, query_id='q', update=0, input_id=None):
    """One authenticatable behavior record from the existing dual contract."""
    logits = logits.detach().float()
    labels = labels.detach().bool()
    probability = torch.sigmoid(logits)
    record = prepare(logits, probability, [f'c{i}' for i in range(len(logits))],
        labels, policy_id='policy', input_id=input_id or 'input-' + query_id,
        query_id=query_id, update=update, algorithm=candidate.BEHAVIOR_ALGORITHM,
        group_size=16, seed=317)
    identity = dict(policy_id='policy', input_id=input_id or 'input-' + query_id,
        query_id=query_id, update=update)
    return record, identity, probability.unsqueeze(0)


def one_gold(count=8, *, query_id='q', update=0):
    logits = torch.linspace(-0.3, 0.3, count)
    labels = torch.zeros(count, dtype=torch.bool)
    labels[count // 2] = True
    return make_record(logits, labels, query_id=query_id, update=update)


def test_generator_rejects_anchor_mismatch(monkeypatch):
    monkeypatch.setattr(candidate, 'formal_source', lambda source_root: 'not the trainer')
    with pytest.raises(ValueError, match='curriculum integration anchor changed'):
        candidate.generate(str(ROOT))


def test_generated_source_compiles_and_installs_the_curriculum_hook():
    source = candidate.generate(str(ROOT))
    compile(source, '<curriculum>', 'exec')
    assert source.count('_dual_actor_module.hybrid_loss=_curriculum_hybrid_loss') == 1
    assert source.count('curriculum_contract(c)') == 1
    assert source.count("algorithm='" + candidate.BEHAVIOR_ALGORITHM + "'") == 2
    assert source.count('commit_diagnostics(out,index)') == 1


def test_schedule_endpoint_loss_coefficients():
    start = candidate.loss_coefficients(0, 212)
    end = candidate.loss_coefficients(212, 212)
    assert start == pytest.approx(dict(supervised=0.5, contrast=0.3, ranking=0.2,
        margin=0.15, brier=0.15, rank=0.2))
    assert end == pytest.approx(dict(supervised=0.0, contrast=0.4, ranking=0.6,
        margin=0.2, brier=0.2, rank=0.6))
    assert schedule(0, 212) == dict(supervised=0.5, contrast=0.3, ranking=0.2)
    assert schedule(212, 212) == dict(supervised=0.0, contrast=0.4, ranking=0.6)


def test_current_and_behavior_log_prob_recompute_the_same_trajectory():
    weights = torch.softmax(torch.linspace(-0.4, 0.4, 8), 0)
    labels = torch.zeros(8, dtype=torch.bool)
    labels[3] = True
    terms = candidate.trajectory_terms(weights, labels,
        schedule_weights=schedule(0, 212), k=8, group_size=16, seed=317,
        update=0, query_id='same-trajectory')
    current = candidate.current_log_prob(weights, terms['draws'])
    assert torch.allclose(terms['log_prob'], current, atol=1e-12, rtol=1e-12)
    assert torch.allclose(terms['log_prob'],
        plackett_luce_log_prob(weights, terms['draws']), atol=1e-12, rtol=1e-12)
    assert bool((terms['log_prob'] <= 0).all())


def test_current_log_prob_keeps_autograd_through_the_shared_draw():
    base = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    draws = torch.tensor([[0, 2, 1, 3]])
    weights = base.clone().requires_grad_(True)
    candidate.current_log_prob(weights, draws).sum().backward()
    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()
    assert float(weights.grad.abs().sum()) > 0
    eps = 1e-4
    numeric = []
    for index in range(len(base)):
        high, low = base.clone(), base.clone()
        high[index] += eps
        low[index] -= eps
        numeric.append(float(candidate.current_log_prob(high, draws)
            - candidate.current_log_prob(low, draws)) / (2 * eps))
    assert torch.allclose(weights.grad, torch.tensor(numeric), atol=1e-3, rtol=1e-3)


@pytest.mark.filterwarnings('ignore:std\\(\\):UserWarning')
def test_zero_advantage_group_keeps_only_supervised_and_kl():
    config = runtime_config()
    candidate.configure(config)
    logits = torch.zeros(8)
    labels = torch.ones(8, dtype=torch.bool)
    record, identity, reference = make_record(logits, labels, query_id='flat')
    total, detail = candidate.hybrid_loss(logits.unsqueeze(0), [record], [identity],
        reference, torch.tensor([True]), global_real_requests=1, world_size=1)
    split = config['constants']['sft_split']
    sft, _ = candidate.supervised_loss(logits, labels,
        nll_weight=split['nll'], infonce_weight=split['infonce'])
    kl = candidate.kl_loss(logits, reference[0])
    expected = 0.5 * sft + 0.01 * kl
    assert detail['rl'] == pytest.approx(0.0, abs=1e-12)
    assert float(total) == pytest.approx(float(expected), rel=1e-6, abs=1e-7)
    assert {'rank', 'brier', 'margin', 'trajectory_log_prob'} <= set(detail['no_signal'])


def test_zero_weight_padding_is_excluded_from_numerator_and_denominator():
    config = runtime_config()
    candidate.configure(config)
    real, real_identity, real_reference = one_gold(query_id='real')
    padding, padding_identity, padding_reference = one_gold(query_id='padding')
    logits = torch.stack([torch.linspace(-0.3, 0.3, 8), torch.full((8,), 100.0)])
    references = torch.cat([real_reference, padding_reference])
    padded, _ = candidate.hybrid_loss(logits, [real, padding],
        [real_identity, padding_identity], references,
        torch.tensor([True, False]), global_real_requests=1, world_size=1)
    alone, _ = candidate.hybrid_loss(logits[:1], [real], [real_identity],
        real_reference, torch.tensor([True]), global_real_requests=1, world_size=1)
    assert torch.allclose(padded, alone)


def test_world_size_and_global_denominator_are_consistent():
    config = runtime_config()
    candidate.configure(config)
    record, identity, reference = one_gold(query_id='denominator')
    logits = torch.linspace(-0.3, 0.3, 8).unsqueeze(0)
    local, _ = candidate.hybrid_loss(logits, [record], [identity], reference,
        torch.tensor([True]), global_real_requests=1, world_size=1)
    candidate.reset_diagnostics()
    replicated, _ = candidate.hybrid_loss(logits, [record], [identity], reference,
        torch.tensor([True]), global_real_requests=2, world_size=2)
    candidate.reset_diagnostics()
    half, _ = candidate.hybrid_loss(logits, [record], [identity], reference,
        torch.tensor([True]), global_real_requests=2, world_size=1)
    assert torch.allclose(local, replicated)
    assert torch.allclose(half, local * 0.5)
    assert float(candidate.global_reduce(torch.tensor(3.0),
        world_size=2, global_real_requests=4)) == pytest.approx(1.5)


def test_trajectory_rejects_k_above_the_menu():
    weights = torch.softmax(torch.linspace(-0.2, 0.2, 4), 0)
    labels = torch.zeros(4, dtype=torch.bool)
    labels[0] = True
    with pytest.raises(ValueError, match='trajectory length inside the menu'):
        candidate.trajectory_terms(weights, labels,
            schedule_weights=schedule(0, 212), k=8, group_size=16, seed=317,
            update=0, query_id='too-long')


def test_no_gold_trajectory_reports_zero_margin_and_reason():
    weights = torch.softmax(torch.linspace(-0.2, 0.2, 8), 0)
    labels = torch.zeros(8, dtype=torch.bool)
    terms = candidate.trajectory_terms(weights, labels,
        schedule_weights=schedule(0, 212), k=8, group_size=16, seed=317,
        update=0, query_id='no-gold')
    assert bool((terms['margin'] == 0).all())
    assert set(terms['margin_reason']) == {'no_gold_in_trajectory'}


def test_supervised_split_comes_from_the_caller_and_contract_rejects_bad_identity():
    logits = torch.linspace(-0.2, 0.2, 8)
    labels = torch.zeros(8, dtype=torch.bool)
    labels[2] = True
    nll_only, detail = candidate.supervised_loss(logits, labels,
        nll_weight=1.0, infonce_weight=0.0)
    assert torch.allclose(nll_only, detail['nll'])
    infonce_only, detail = candidate.supervised_loss(logits, labels,
        nll_weight=0.0, infonce_weight=1.0)
    assert torch.allclose(infonce_only, detail['infonce'])
    bad = runtime_config()
    bad['algorithm'] = candidate.BEHAVIOR_ALGORITHM
    with pytest.raises(ValueError, match='fresh formal curriculum identity'):
        candidate.contract(bad)


def test_contract_accepts_the_preregistered_runtime_identity():
    formal = candidate.contract(runtime_config())
    assert formal['updates'] == 212
    assert formal['recovery_checkpoint_steps'] == DESIGN['evaluation']['recovery_checkpoint_steps']
    assert formal['curriculum_algorithm'] == candidate.CURRICULUM_ALGORITHM
    assert formal['behavior_algorithm'] == candidate.BEHAVIOR_ALGORITHM


def test_topology_delegates_the_behavior_identity_to_the_frozen_layout():
    config = runtime_config()
    config.update(
        physical_devices=[0, 1, 2, 3, 4, 5],
        macro_client=dict(schema='jev-replica-macro-client-v1', max_queries=384,
            rollout_queries=32, replicas=[{}, {}], runtime_identity='runtime'),
        dual_runtime_identity='runtime', global_batch=384, microbatch=8)
    layout = candidate.topology(config, 6, [f'davinci{i}' for i in range(6)])
    assert layout['world_size'] == 6
    assert layout['global_batch'] == 384
    assert layout['total_updates'] == 212


def test_diagnostics_accumulate_micro_batches_and_write_json(tmp_path):
    candidate.configure(runtime_config())
    record, identity, reference = one_gold(query_id='diagnostics')
    logits = torch.linspace(-0.3, 0.3, 8).unsqueeze(0)
    candidate.hybrid_loss(logits, [record], [identity], reference,
        torch.tensor([True]), global_real_requests=1, world_size=1)
    candidate.hybrid_loss(logits, [record], [identity], reference,
        torch.tensor([True]), global_real_requests=1, world_size=1)
    candidate.commit_diagnostics(tmp_path, 0)
    payload = json.loads((tmp_path / 'curriculum-0000.json').read_text())
    assert payload['micro_batches'] == 2
    assert isinstance(payload['no_signal'], list)
    assert payload['schedule'] == dict(supervised=0.5, contrast=0.3, ranking=0.2)

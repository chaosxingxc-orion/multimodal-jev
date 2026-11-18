import ast
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from multimodal_jev.context_candidate_objective import loss as auxiliary
from tools.reports.context_loss_metrics_candidate import contributions, generate, validate_total, PREFIX

ACTOR = Path('.local/decision-objective-research/context-rl-grpo-arithmetic-v2-package/source/src/multimodal_jev/context_native_actor.py')


def function(source, objective):
    # Extract the actual registered_loss body. No fake verl import or native
    # execution claim: sampled terms below are a differentiable test double.
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'registered_loss')
    namespace = dict(torch=torch, dist=NS(get_world_size=lambda: 4), MODE='test',
                     _active=ContextVar('test'), loss=objective,
                     contributions=contributions, validate_total=validate_total)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<registered-loss-test>', 'exec'), namespace)
    return namespace


def objective(z, y, mask, weights, real, **kw):
    aux, metrics = auxiliary(z, y, mask, weights, real,
        global_real_requests=kw['global_real_requests'], world_size=kw['world_size'],
        ranking_coefficient=kw['ranking_coefficient'], probability_coefficient=kw['probability_coefficient'])
    detail = dict(stage=kw['stage'], auxiliary=metrics)
    if kw['stage'] == 'instruction':
        return aux, detail
    scale = kw['world_size'] / kw['global_real_requests']
    # Stand-in sampled terms exercise accounting and gradient preservation;
    # this intentionally does not test the native GRPO surrogate.
    # The admitted probability arithmetic and sampled objective are FP32 even
    # when execution logits are BF16; preserve that contract in this double.
    z32 = z.float()
    rl = (z32.sum(-1) * weights).sum() * scale * .017
    kl = (z32.square().sum(-1) * weights).sum() * scale
    detail['sampled'] = dict(rl=float(rl.detach()), kl=float(kl.detach()),
        coefficient=0., kl_coefficient=kw['kl_coefficient'], global_real_requests=kw['global_real_requests'])
    return aux + rl + kw['kl_coefficient'] * kl, detail


def invoke(namespace, z, stage):
    payload = []
    for i in range(2):
        payload.append(dict(labels=torch.tensor([[True, False, False]]),
            observed=torch.tensor([[True, True, i == 0]]), request_weight=1. if i == 0 else 0.,
            real_request=i == 0, global_real_requests=7, stage=stage,
            ranking_coefficient=.2, probability_coefficient=.3,
            actions=None, old_joint=None, advantages=None, reference_policy=None,
            kl_coefficient=.01 if stage == 'task' else 0.))
    state = dict(consumed=False, log_probs=z, logits=z, payload=payload)
    token = namespace['_active'].set(state)
    try:
        config = NS(policy_loss=NS(loss_mode='test'), clip_ratio=.2,
                    ppo_mini_batch_size=32, ppo_micro_batch_size_per_gpu=8)
        return namespace['registered_loss'](None, z, None, None, config=config)
    finally:
        namespace['_active'].reset(token)


@pytest.mark.parametrize('stage', ['instruction', 'task'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_candidate_keeps_exact_loss_gradient_and_one_adam_update(stage, dtype):
    candidate, _ = generate(ACTOR)
    original = function(ACTOR.read_text(), objective)
    modified = function(candidate, objective)
    results = []
    for module in (original, modified):
        parameter = torch.nn.Parameter(torch.tensor([[.2, -.4, .3], [.1, .2, -.3]]))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4, foreach=False, fused=False)
        value, metrics = invoke(module, parameter.to(dtype), stage)
        (value / 4).backward()  # Native actor accumulation cancellation.
        gradient = parameter.grad.clone()
        optimizer.step()
        results.append((value.detach(), gradient, parameter.detach().clone(), metrics))
    for i in range(3):
        assert torch.equal(results[0][i], results[1][i])
    validate_total(results[1][3])
    assert torch.equal(results[1][1][1], torch.zeros(3))


def test_sum_microsteps_then_mean_ranks_not_mean_microsteps():
    rows = []
    for rank in range(4):
        micro = []
        for step in range(4):
            payload = dict(stage='task', global_real_requests=128,
                           ranking_coefficient=.3, probability_coefficient=.2, kl_coefficient=.01)
            detail = dict(stage='task', auxiliary=dict(global_real_requests=128,
                ranking_numerator=rank + step + 1., probability_numerator=2.),
                sampled=dict(rl=-.02, kl=.5, coefficient=0., kl_coefficient=.01, global_real_requests=128))
            micro.append(contributions(detail, payload, 4))
        rows.append(micro)
    total = sum(sum(x[PREFIX+'reconstructed_total'] for x in row) for row in rows) / 4
    expected = .3 * sum(rank + step + 1 for rank in range(4) for step in range(4)) / 128
    expected += .2 * 32 / 128 + 4 * (-.02 + .01 * .5)
    assert total == pytest.approx(expected)
    assert total / 4 != pytest.approx(expected)


def test_tampered_actor_identity_rejected(tmp_path):
    path = tmp_path/'actor.py'; path.write_bytes(ACTOR.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='identity'):
        generate(path)


def test_bad_component_total_rejected():
    metrics = {PREFIX+k: 0. for k in ('rl','ranking_weighted','probability_weighted','kl_weighted','reconstructed_total')}
    metrics['jev/context_loss'] = .1
    with pytest.raises(ValueError, match='component sum'):
        validate_total(metrics)


def test_wrong_denominator_or_hidden_ce_rejected():
    payload = dict(stage='task', global_real_requests=128, ranking_coefficient=.3,
                   probability_coefficient=.3, kl_coefficient=.01)
    detail = dict(stage='task', auxiliary=dict(global_real_requests=127,
        ranking_numerator=1., probability_numerator=1.))
    with pytest.raises(ValueError, match='denominator'):
        contributions(detail, payload, 4)
    detail['auxiliary']['global_real_requests'] = 128
    detail['sampled'] = dict(global_real_requests=128, coefficient=.1, kl_coefficient=.01)
    with pytest.raises(ValueError, match='contract'):
        contributions(detail, payload, 4)

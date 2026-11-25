import json
from pathlib import Path

import pytest
import torch

from multimodal_jev.card_experiment_config import adamw, bind_stage1, prior_runtime
from multimodal_jev.stage1_card_layout import CardLayout
from tools.reports.build_card_tpo_targets import target_rank_counts
from tools.reports.card_tower_comparison import comparison_policy
from tools.reports.prepare_card_scoring_benchmark import ssh_limits
from tools.reports.prepare_paired_card_evaluation import history_result_key


def test_bound_settings_affect_adam_and_reject_changed_resume():
    custom = dict(seed=13, stage1={'grad_clip': .4, 'optimizer': {'betas': [.6, .8], 'weight_decay': .2}})
    train = bind_stage1({}, custom)
    rt = train['stage1_runtime']
    assert rt['actor_seed'] == 13
    p = torch.nn.Parameter(torch.tensor(1.))
    opt = adamw([p], rt, .1)
    p.grad = torch.tensor(2.)
    opt.step()
    assert p.item() == pytest.approx(.88)
    assert opt.state[p]['exp_avg'].item() == pytest.approx(.8)
    assert bind_stage1(train, {}) == train
    with pytest.raises(ValueError, match='bound parent'):
        bind_stage1(train, {'stage1': {'grad_clip': .5}})
    with pytest.raises(ValueError, match='unknown'):
        bind_stage1({}, {'stage1': {'grad_clp': 1.}})


def test_registered_legacy_runtime_is_an_independent_pin():
    ops = Path('configs/portable/operations/card-stage1-tower-m16-v1.json')
    origin = json.loads(ops.read_bytes())['window']['continuation']
    pins = json.loads(Path('configs/experiments/card-legacy-stage1-runtime-v1.json').read_bytes())
    origin = dict(origin, legacy_runtime_sha256=pins[origin['formal_sha256']])
    runtime = bind_stage1({}, json.loads(ops.read_bytes()))['stage1_runtime']
    assert prior_runtime({}, runtime, origin)['stage1_runtime'] == runtime
    with pytest.raises(ValueError, match='legacy'):
        prior_runtime({}, dict(runtime, actor_seed=0), origin)
    with pytest.raises(ValueError, match='legacy'):
        prior_runtime({}, runtime, {})


def test_epoch_assignment_and_all_epoch_target_audit():
    panel = dict(questions=[dict(id='a', definition='Intent A', question='A?')],
                 requirements=['listen'], history=[], version='jev-audio-context-applicability-v1')
    plan = dict(assignments={'u': ['ex', 'rules', 'ex']}, panels=dict(rules=panel, ex=panel))
    layout = CardLayout(plan, ['u'], dict(schema={'candidates':[{'id':'a'}]}), epochs=3)
    assert layout('audio', 'u', 2).audio == 'audio'
    with pytest.raises(ValueError, match='assignment'):
        CardLayout(plan, ['u'], dict(schema={'candidates':[{'id':'a'}]}), epochs=2)
    assert target_rank_counts(torch.tensor([[[3., 1.]], [[1., 2.]], [[0., 0.]]]), torch.tensor([0])) == {
        'strict': 1, 'wrong': 1, 'tied': 1}


def test_packaging_honors_custom_spans_timeout_and_baseline():
    c = dict(spans=[{'name':'short'}, {'name':'stress'}], logit_atol=0., max_time_ratio=.98,
             tower_comparison=dict(measured_spans=['stress'], pressure_span='stress', quantile=.75))
    assert comparison_policy(c)['quantile'] == .75
    with pytest.raises(ValueError, match='spans'):
        comparison_policy(dict(c, spans=[{'name':'short'}]))
    cmd, timeout = ssh_limits({'scoring_ssh': {'connect_seconds': 6, 'command_seconds': 17}})
    assert 'ConnectTimeout=6' in cmd and timeout == 17
    assert history_result_key('JEV-208') == 'versus_208'


def test_reference_loader_consumes_bound_seed_and_threads(monkeypatch):
    import sys
    from types import SimpleNamespace
    from tools.reports import context_transfer_macro_parity as reference
    from tools.jev import context_configurable_device_scope as scope

    calls = {}
    def record(key):
        return lambda value: calls.update({key:value})
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace())
    monkeypatch.setattr(scope, 'validate_tp1', lambda cfg: None)
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(set_device=record('device'), manual_seed_all=record('npu_seed')), raising=False)
    monkeypatch.setattr(torch, 'set_num_threads', record('threads'))
    monkeypatch.setattr(torch, 'manual_seed', record('seed'))
    monkeypatch.setattr(reference, 'load_native_reference', lambda *args: None)
    monkeypatch.setattr(torch, 'device', lambda *args: args)
    reference.load_reference_model(dict(physical_devices=[7], process_device_scope={},
        reference_inference={'seed':12,'torch_threads':3}), {}, None)
    assert calls == {'threads':3,'seed':12,'npu_seed':12,'device':0}

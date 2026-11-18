"""Real optimizer continuation; source admission composition has explicit substitutes."""
import copy
import json
from pathlib import Path
import pytest
torch = pytest.importorskip('torch')
from multimodal_jev import simple_stage2_resume as module
from multimodal_jev.baseline import file_hash


def tensors():
    from test_span_fsdp_resume import fixture
    original, old_optimizer, loaded, fresh, state = fixture()
    # Existing fixture has actual sub-BF16 FP32 masters and nonzero Adam.
    state.update(stage='simple_stage2_rl', next_update=state['updates'], algorithm='grpo_v1',
                 reference_policy_id='base:stage1', adapter_id='rl', schedule_sha256='schedule')
    recovered = dict(states=[state], completed_updates=state['updates'], stage='simple_stage2_rl',
        policy_id='base:rl', adapter_id='rl', reference_policy_id='base:stage1',
        initial=dict(reference_policy_id='base:stage1'))
    return original, old_optimizer, loaded, fresh, recovered


def plan():
    return dict(mode='final_stage1_fresh_optimizer', start_update=0, stop_update=634,
                total_updates=634, reset_optimizer=True)


def test_recovered_master_adam_next_update_matches_uninterrupted():
    original, old_optimizer, loaded, fresh, recovered = tensors()
    loaded.weight.grad = torch.ones_like(loaded.weight)
    actual, receipt = module.restore(loaded, fresh, recovered, rank=0)
    assert receipt['exact_state']['exact'] and loaded.weight.grad is None
    assert actual['optimizer']['state']
    result = module.continuous_plan(plan(), recovered, fresh, base_id='base', adapter_id='rl')
    assert result['start_update'] == recovered['completed_updates'] and result['stop_update'] == 634
    for model, optimizer in ((original, old_optimizer), (loaded, fresh)):
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(1, 3)).square().sum().backward(); optimizer.step()
    assert torch.equal(original.weight, loaded.weight)
    for name, value in old_optimizer.state_dict()['state'][0].items():
        assert torch.equal(value, fresh.state_dict()['state'][0][name])


@pytest.mark.parametrize('fault', ['reset', 'cursor', 'reference', 'policy', 'stage', 'gradient', 'stop'])
def test_recovery_cannot_restart_optimizer_reference_or_schedule(fault):
    _, _, loaded, fresh, recovered = tensors()
    module.restore(loaded, fresh, recovered, rank=0); p = plan()
    if fault == 'reset': fresh.state.clear()
    elif fault == 'cursor': recovered['completed_updates'] += 1
    elif fault == 'reference': recovered['reference_policy_id'] = 'base:rl'
    elif fault == 'policy': recovered['policy_id'] = 'base:stage1'
    elif fault == 'stage': recovered['stage'] = 'continuous_rl_engineering'
    elif fault == 'gradient': loaded.weight.grad = torch.ones_like(loaded.weight)
    else: p['stop_update'] = 3
    with pytest.raises(ValueError): module.continuous_plan(p, recovered, fresh, base_id='base', adapter_id='rl')


def admission(tmp_path, monkeypatch):
    import multimodal_jev.simple_stage2_release as release
    import multimodal_jev.simple_stage2_initialization as initial
    import multimodal_jev.simple_continuous_resume as resume
    path = tmp_path / 'run/evaluation/selection-0000/result.json'
    path.parent.mkdir(parents=True); path.write_text('{}')
    digest = file_hash(path)
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps(dict(selection=[dict(passed=True, updates=0, policy_id='base:stage1',result=str(path),
                                                      result_sha256=digest)])))
    recovered = dict(reference_policy_id='base:stage1',completed_updates=3,files={str(path):digest})
    calls=[]
    def released(config): calls.append('release'); return dict(files={})
    def initialized(config,*,verify_parity): calls.append('initial'); return dict(reference_policy_id='base:stage1',files={})
    def resumed(config,*,formal):
        assert formal is True; calls.append('resume'); return copy.deepcopy(recovered)
    monkeypatch.setattr(release,'authenticate',released)
    monkeypatch.setattr(initial,'authenticate',initialized)
    monkeypatch.setattr(resume,'authenticate',resumed)
    config=dict(continuous_resume=dict(training_root=str(tmp_path),audit=str(audit)))
    return config,recovered,path,audit,calls


def test_release_initial_reference_and_full_recovery_all_required(tmp_path,monkeypatch):
    config,_,path,_,calls=admission(tmp_path,monkeypatch)
    result=module.authenticate(config,verify_parity=None)
    assert calls==['release','initial','resume']
    assert result['selection_history']==[dict(updates=0,policy_id='base:stage1',
        result=str(path),result_sha256=file_hash(path))]


@pytest.mark.parametrize('fault',['reference','unbound_selection','changed_selection','audit_hash','unaccepted'])
def test_join_refuses_wrong_reference_or_unverified_selection_history(tmp_path,monkeypatch,fault):
    config,recovered,path,audit,_=admission(tmp_path,monkeypatch)
    if fault=='reference': recovered['reference_policy_id']='base:rl'
    elif fault=='unbound_selection': recovered['files'].clear()
    elif fault=='changed_selection':path.write_text('changed')
    else:
        value=json.loads(audit.read_text())
        value['selection'][0]['result_sha256' if fault=='audit_hash' else 'passed']='wrong'
        audit.write_text(json.dumps(value))
    with pytest.raises(ValueError):module.authenticate(config,verify_parity=None)

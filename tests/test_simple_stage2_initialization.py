import copy
import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
from multimodal_jev.baseline import file_hash
from multimodal_jev.resident_service_admission import IMAGE
from multimodal_jev.simple_stage2_initialization import authenticate, restore_stage_boundary, continuous_plan
from multimodal_jev.slurp_instruction_resume import FIELDS


def source(tmp_path):
    """Real file graph; independent parity computation has separate full tests."""
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    root, gate, adapter = [tmp_path / x for x in ('stage1', 'gate', 'adapter')]
    common = {key: 'fixed' for key in FIELDS}
    common.update(total_updates=634, task_mode='direct_final', loss_reduction='real_request_mean',
                  microbatch_per_rank=1)
    put(root / 'config.json', common)
    put(root / 'launch.json', dict(image=IMAGE, container_id='stage1'))
    paths = [root / 'run' / f'rank-{r}/checkpoint-0634.pt' for r in range(4)]
    paths += [adapter / n for n in ('adapter_config.json', 'adapter_model.safetensors')]
    for i, path in enumerate(paths):
        put(path, dict(asset=i))
    state_path = tmp_path / 'audit/state-verified.json'
    state = dict(passed=True, updates=634, epoch=2, next_batch=0, ranks=4, tensors=112,
        exact_master_export=True, exact_peft_reload=True, durable_commit_verified=True,
        training_root=str(root), adapter=str(adapter), adapter_id='accepted',
        files={str(p): file_hash(p) for p in paths})
    put(state_path, state)
    put(gate / 'config.json', dict(checkpoint_evidence=str(state_path)))
    put(gate / 'launch.json', dict(image=IMAGE, container_id='selection',
                                 config_sha256=file_hash(gate / 'config.json')))
    put(gate / 'observed-final.json', dict(image=IMAGE, container_id='selection',
                                         state=dict(Status='exited', ExitCode=0)))
    files = {str(p): file_hash(p) for p in tmp_path.rglob('*') if p.is_file()}
    result = dict(passed=True, full_candidate_parity=True, exact_master_export=True,
        updates=634, requests=9226, population_mode='full_selection', training_root=str(root),
        state_sha256=file_hash(state_path), adapter=str(adapter), adapter_id='accepted', files=files)
    config = dict(common, stage2_initialization=dict(mode='final_stage1_fresh_optimizer',
        algorithm='grpo_v1', start_update=0, total_updates=634, reset_optimizer=True, gate=str(gate)),
        pins={str(p): file_hash(p) for p in gate.iterdir()})
    def verify(path):
        assert Path(path) == gate
        return result
    return config, result, verify, put


def test_final_stage1_becomes_initial_and_fixed_reference_with_new_cursor(tmp_path):
    config, _, verify, _ = source(tmp_path)
    report = authenticate(config, verify_parity=verify)
    assert report['source_updates'] == 634 and report['completed_updates'] == report['next_update'] == 0
    assert report['policy_id'] == report['reference_policy_id'] == 'fixed:accepted'
    assert len(report['shard_paths']) == 4


def test_historical_stage1_implicit_microbatch_one_is_preserved(tmp_path):
    config, gate, verify, put = source(tmp_path)
    path = Path(gate['training_root']) / 'config.json'
    value = json.loads(path.read_text()); value.pop('microbatch_per_rank'); put(path, value)
    gate['files'][str(path)] = file_hash(path)
    assert authenticate(config, verify_parity=verify)['completed_updates'] == 0
    config['microbatch_per_rank'] = 2
    with pytest.raises(ValueError, match='microbatch'):
        authenticate(config, verify_parity=verify)


@pytest.mark.parametrize('fault', ['prefix', 'partial_selection', 'live_gate', 'gate_pin',
    'wrong_image', 'missing_shard', 'missing_asset', 'altered_shard', 'unverified_state',
    'unfinished_epoch', 'rl_source', 'config_change', 'bool_cursor', 'nonzero_cursor', 'no_reset'])
def test_reject_partial_changed_or_non_stage1_sources(tmp_path, fault):
    config, gate, verify, put = source(tmp_path)
    root = Path(gate['training_root'])
    if fault == 'prefix': gate['updates'] = 2
    elif fault == 'partial_selection': gate['requests'] = 256
    elif fault in ('bool_cursor', 'nonzero_cursor'):
        config['stage2_initialization']['start_update'] = False if fault == 'bool_cursor' else 1
    elif fault == 'no_reset': config['stage2_initialization']['reset_optimizer'] = False
    elif fault == 'config_change': config['lora_rank'] = 99
    elif fault == 'altered_shard':
        (root / 'run/rank-0/checkpoint-0634.pt').write_text('altered')
    elif fault in ('missing_shard', 'missing_asset'):
        path = root / 'run/rank-0/checkpoint-0634.pt' if fault == 'missing_shard' else Path(gate['adapter']) / 'adapter_model.safetensors'
        gate['files'].pop(str(path))
    else:
        if fault in ('live_gate', 'wrong_image', 'gate_pin'):
            path = Path(config['stage2_initialization']['gate']) / 'observed-final.json'
        elif fault == 'rl_source': path = root / 'config.json'
        else: path = tmp_path / 'audit/state-verified.json'
        value = json.loads(path.read_text())
        if fault == 'live_gate': value['state']['Status'] = 'running'
        elif fault == 'wrong_image': value['image'] = 'other'
        elif fault == 'gate_pin': value['extra'] = True
        elif fault == 'rl_source': value['sampled_probe'] = {}
        elif fault == 'unverified_state': value['exact_peft_reload'] = False
        else: value['epoch'] = 1
        put(path, value)
        gate['files'][str(path)] = file_hash(path)
        if path.name == 'state-verified.json': gate['state_sha256'] = file_hash(path)
        if str(path) in config['pins'] and fault != 'gate_pin': config['pins'][str(path)] = file_hash(path)
    with pytest.raises(ValueError): authenticate(config, verify_parity=verify)


def tensors():
    from test_span_fsdp_resume import fixture
    original, old_optimizer, loaded, fresh, state = fixture()
    state['updates'] = 634
    for adam in state['optimizer']['state'].values(): adam['step'].fill_(634)
    return original, loaded, fresh, state


def test_stage_boundary_keeps_exact_masters_resets_adam_and_matches_fresh_next_step():
    original, model, optimizer, state = tensors()
    assert not torch.equal(original.weight, model.weight)
    # A stale gradient must not leak across stage boundaries.
    model.weight.grad = torch.ones_like(model.weight)
    report = restore_stage_boundary(model, optimizer, state, rank=0)
    assert report['optimizer_reset'] and report['source_adam_validated']
    assert torch.equal(model.weight, original.weight) and not optimizer.state
    assert model.weight.grad is None
    reference = copy.deepcopy(original)
    ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-4, foreach=False)
    for m, opt in [(model, optimizer), (reference, ref_optimizer)]:
        m(torch.ones(1, 3)).square().sum().backward(); opt.step()
    assert torch.equal(model.weight, reference.weight)
    for key, value in optimizer.state_dict()['state'][0].items():
        assert torch.equal(value, ref_optimizer.state_dict()['state'][0][key])
    assert optimizer.state_dict()['state'][0]['step'].item() == 1


@pytest.mark.parametrize('fault', ['cursor', 'nan', 'wrong_adapter'])
def test_stage_boundary_failure_preserves_actor_and_fresh_optimizer(fault):
    _, model, optimizer, state = tensors()
    if fault == 'cursor': state['updates'] = 2
    elif fault == 'nan': state['optimizer']['state'][0]['exp_avg'][0, 0] = float('nan')
    else: state['parameters']['weight'] += 1
    before = model.weight.detach().clone()
    with pytest.raises(ValueError): restore_stage_boundary(model, optimizer, state, rank=0)
    assert torch.equal(before, model.weight) and not optimizer.state


def test_initialization_invokes_real_full_selection_parity_verifier(tmp_path, monkeypatch):
    """Recompute all 9226 input/probability bindings, not a passed-flag stub.

    Model identity and released input compilation use the parity fixture's
    substitutes; tensor state reset is tested separately above.
    """
    from test_slurp_simple_parity_verifier import selection_fixture, m
    gate_root, put = selection_fixture(tmp_path, monkeypatch)
    root = tmp_path / 'train'
    old = json.loads((root / 'config.json').read_text())
    for key in FIELDS: old.setdefault(key, 'fixed')
    old.update(microbatch_per_rank=1, loss_reduction='real_request_mean')
    put(root / 'config.json', old)
    for folder, launch_name, observed_name in [
        (root, 'launch.json', None), (gate_root, 'launch.json', 'observed-final.json')]:
        launch = json.loads((folder / launch_name).read_text())
        launch.update(image=IMAGE, config_sha256=file_hash(folder / 'config.json'))
        put(folder / launch_name, launch)
        if observed_name:
            observed = json.loads((folder / observed_name).read_text()); observed['image'] = IMAGE
            put(folder / observed_name, observed)
    observed = json.loads((gate_root / 'training-observed.json').read_text()); observed['image'] = IMAGE
    put(gate_root / 'training-observed.json', observed)
    state_path = gate_root / 'state-verified.json'
    state = json.loads(state_path.read_text()); state.update(ranks=4, tensors=112)
    assets = [root / 'run' / f'rank-{r}/checkpoint-0634.pt' for r in range(4)]
    assets += [Path(state['adapter']) / n for n in ('adapter_config.json', 'adapter_model.safetensors')]
    for i, path in enumerate(assets): put(path, dict(asset=i))
    state['files'] = {str(p): file_hash(p) for p in assets}
    put(state_path, state)
    gate_config = json.loads((gate_root / 'config.json').read_text())
    gate_config['pins'].update({str(p): file_hash(p) for p in
                              [state_path, gate_root / 'training-observed.json']})
    put(gate_root / 'config.json', gate_config)
    launch = json.loads((gate_root / 'launch.json').read_text())
    launch['config_sha256'] = file_hash(gate_root / 'config.json'); put(gate_root / 'launch.json', launch)
    config = dict(old, stage2_initialization=dict(mode='final_stage1_fresh_optimizer',
        algorithm='grpo_v1', start_update=0, total_updates=634, reset_optimizer=True, gate=str(gate_root)),
        pins={str(gate_root / n): file_hash(gate_root / n) for n in
              ('config.json', 'launch.json', 'observed-final.json')})
    report = authenticate(config, verify_parity=m.verify)
    assert report['reference_policy_id'] == 'b:a'
    # A forged success flag cannot mask a changed last-request distribution.
    path = gate_root / 'run/vllm/result.json'; value = json.loads(path.read_text())
    value['records'][-1]['candidate_logprobs'][0][0] += 1
    put(path, value)
    with pytest.raises(ValueError): authenticate(config, verify_parity=m.verify)


@pytest.mark.parametrize('algorithm', ['grpo_v1', 'rloo_v1'])
def test_stage2_loop_plan_uses_new_full_schedule_and_same_fixed_reference(tmp_path, algorithm):
    config, _, verify, _ = source(tmp_path)
    plan = config['stage2_initialization']; plan.update(algorithm=algorithm, stop_update=634)
    initialized = authenticate(config, verify_parity=verify)
    _, model, optimizer, state = tensors()
    restore_stage_boundary(model, optimizer, state, rank=0)
    result = continuous_plan(plan, initialized, optimizer, base_id='fixed', adapter_id='accepted')
    assert result == dict(policy_id='fixed:accepted', reference_policy_id='fixed:accepted',
                          start_update=0, stop_update=634)
    # Neither a leftover optimizer history nor a truncated fixture budget is a
    # valid substitute for the declared full-data stage.
    for altered in (dict(plan, stop_update=3), dict(plan, start_update=1), dict(plan, start_update=False)):
        with pytest.raises(ValueError):
            continuous_plan(altered, initialized, optimizer, base_id='fixed', adapter_id='accepted')
    model(torch.ones(1, 3)).square().sum().backward(); optimizer.step(); optimizer.zero_grad()
    with pytest.raises(ValueError):
        continuous_plan(plan, initialized, optimizer, base_id='fixed', adapter_id='accepted')

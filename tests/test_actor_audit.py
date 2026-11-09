import copy
import pytest
from multimodal_jev.actor_audit import validate_state

torch = pytest.importorskip('torch')


def fixture(updates=2):
    weights = {'layer.lora_A.weight': torch.ones(2, 3)}
    config = {'lr': 1e-5, 'weight_decay': 0., 'foreach': False, 'fused': False}
    moments = {0: {'step': torch.tensor(float(updates)),
                   'exp_avg': torch.zeros(2, 3), 'exp_avg_sq': torch.ones(2, 3)}} if updates else {}
    state = {'parameter_names': ['layer.lora_A.default.weight'], 'parameter_shapes': [[2, 3]],
             'optimizer': {'state': moments, 'param_groups': [{**config, 'params': [0]}]},
             'cpu_rng': torch.get_rng_state()}
    return state, weights, config


@pytest.mark.parametrize('updates', [0, 1, 2])
def test_saved_adamw_counter_and_mapping(updates):
    state, weights, config = fixture(updates)
    before = copy.deepcopy(state)
    report = validate_state(state, weights, updates, config)
    assert report['optimizer_step'] == updates
    assert report['optimizer_states'] == int(updates > 0)
    assert state['parameter_names'] == before['parameter_names']
    assert torch.equal(state['cpu_rng'], before['cpu_rng'])


@pytest.mark.parametrize('mutation,match', [
    (lambda s: s['optimizer']['state'][0].update(step=torch.tensor(1.)), 'step differs'),
    (lambda s: s['optimizer']['state'].clear(), 'coverage'),
    (lambda s: s['optimizer']['state'][0].update(exp_avg=torch.full((2,3),float('nan'))), 'finite'),
    (lambda s: s['optimizer']['state'][0].update(exp_avg_sq=-torch.ones(2,3)), 'negative'),
    (lambda s: s['optimizer']['param_groups'][0].update(lr=1e-4), 'configuration'),
    (lambda s: s.update(parameter_names=['other']), 'mapping'),
    (lambda s: s.update(parameter_shapes=[[3,2]]), 'shape'),
    (lambda s: s.update(cpu_rng=torch.zeros(4)), 'RNG'),
])
def test_bad_saved_state_is_rejected(mutation, match):
    state, weights, config = fixture()
    mutation(state)
    with pytest.raises(ValueError, match=match):
        validate_state(state, weights, 2, config)


def test_nonfinite_adapter_rejected():
    state, weights, config = fixture()
    weights['layer.lora_A.weight'][0,0] = float('inf')
    with pytest.raises(ValueError, match='finite'):
        validate_state(state, weights, 2, config)


@pytest.mark.parametrize('corrupt', [False, True])
def test_parent_noop_must_preserve_saved_moments(tmp_path, monkeypatch, corrupt):
    # Real torch serialization, with a tiny in-memory adapter loader in place of
    # optional safetensors. This exercises lineage; it is not model training.
    import json
    import sys
    import types
    from multimodal_jev.actor_audit import audit_actor
    from multimodal_jev.baseline import file_hash
    tensors = {}
    fake = types.ModuleType('safetensors.torch')
    fake.load_file = lambda path, **kw: tensors[path]
    monkeypatch.setitem(sys.modules, 'safetensors', types.ModuleType('safetensors'))
    monkeypatch.setitem(sys.modules, 'safetensors.torch', fake)
    initial = tmp_path/'initial';initial.mkdir()
    (initial/'adapter_model.safetensors').write_bytes(b'initial')
    tensors[str(initial/'adapter_model.safetensors')] = {'layer.lora_A.weight': torch.zeros(2,3)}
    parent = tmp_path/'parent';child = tmp_path/'child'
    for run, ref, updated, iteration in [(parent,initial,True,1),(child,parent/'adapter',False,2)]:
        (run/'adapter').mkdir(parents=True)
        weight_path = run/'adapter/adapter_model.safetensors'
        weight_path.write_bytes(b'updated')
        state, weights, config = fixture(1)
        tensors[str(weight_path)] = weights
        if run == child and corrupt:
            state['optimizer']['state'][0]['exp_avg'][0,0] = 1.
        torch.save(state, run/'optimizer.pt')
        result = {'status':'passed', 'updates':1, 'iteration':iteration,
                  'adapter_files': {'adapter_model.safetensors':file_hash(weight_path)},
                  'optimizer_sha256':file_hash(run/'optimizer.pt'), 'optimizer_config':config,
                  'base_policy_id':'base', 'reward_mode':'grpo', 'pair_bonus_weight':0.,
                  'config': {'initial_adapter':str(initial) if run == parent else None,
                             'parent_run':str(parent) if run == child else None,
                             'max_logprob_error':.02},
                  'metrics': {'updated':updated, 'max_logprob_error':0.}}
        (run/'result.json').write_text(json.dumps(result))
        (run/'terminal.json').write_text('{"exit_code":0}')
        (run/'identity.json').write_text(json.dumps({'adapter_files':{
            p.name:file_hash(p) for p in ref.iterdir() if p.is_file()}}))
    if corrupt:
        with pytest.raises(ValueError, match='no-op changed optimizer moments'):
            audit_actor(child)
    else:
        report = audit_actor(child)
        assert report['optimizer_step'] == 1 and report['iteration'] == 2
        assert report['changed_adapter_tensors'] == 0 and not report['updated']

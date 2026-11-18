"""Real identity/sampling joins; audio compiler and engine I/O are substitutes."""
import copy
import pytest
torch = pytest.importorskip('torch')
from multimodal_jev.simple_resident_batch import ResidentSimpleBatch
from test_simple_audio_batch import fixture


def setup(monkeypatch):
    population, slots, processor, audio_calls = fixture(monkeypatch)
    calls = []
    class Engine:
        def rollout(self, config, renderer):
            assert config['population']['requests'] == len(renderer)
            assert all(set(row[0]) == {'id'} for row in renderer)
            assert all(set(row[5]['request']) == {'audio', 'questions'} for row in renderer)
            calls.append(config['adapter_id'])
            p = torch.full((60,), .5/59)
            p[2 if config['adapter_id'] == 'reference' else 7] = .5
            return dict(passed=True, backend='vllm', base_id='base', adapter_id=config['adapter_id'],
                records=[dict(request_id=r[4], request_identity=r[5], candidate_logprobs=[p.log().tolist()])
                         for r in renderer])
    config = dict(model='model', base_id='base', adapter='initial', adapter_id='reference',
                  runtime_profile='fixed', max_model_len=4096, physical_devices=[0])
    collector = ResidentSimpleBatch(Engine(), population, {}, processor, config, seed=317)
    return collector, population, slots, config, calls, audio_calls


def test_stage_boundary_and_next_batch_use_one_compile_and_same_engine(monkeypatch):
    c, population, slots, config, calls, audio = setup(monkeypatch)
    # Final epoch layout: all 128 slots are retained, including 63 zero padding.
    slots[65:] = [dict(slots[0], padding=True, weight=0.) for _ in range(63)]
    initial, receipt = c.collect(slots, policy_config=config, completed_updates=0)
    assert len(audio) == 65 and calls == ['reference']
    assert receipt['engine_passes'] == 1 and receipt['sampling']['global_real_requests'] == 65
    assert len(initial) == len({x['request_id'] for x in initial}) == 128
    assert all(not x['real_request'] and x['request_weight'] == 0 for x in initial[65:])
    updated = dict(config, adapter='next', adapter_id='updated')
    items, receipt = c.collect(slots, policy_config=updated, completed_updates=1)
    assert len(audio) == 130 and calls == ['reference', 'reference', 'updated']
    assert receipt['next_optimizer_update'] == 2 and receipt['independent_hf_passes'] == 0
    assert all(x['behavior'][0].argmax().item() == 7 for x in items)
    assert all(x['reference_policy'][0].argmax().item() == 2 for x in items)
    # Tiny FP32 updates may export identical BF16: do not demand new bytes.
    c.collect(slots, policy_config=updated, completed_updates=2)
    assert c.last_update == 2


def test_labels_do_not_change_rollout_or_sampling(monkeypatch):
    c, population, slots, config, calls, audio = setup(monkeypatch)
    before, receipt = c.collect(slots, policy_config=config, completed_updates=0)
    for row in population.values(): row['target'] = (row['target']+1)%60
    other = ResidentSimpleBatch(c.engine, population, {}, c.processor, config, seed=317)
    after, second = other.collect(slots, policy_config=config, completed_updates=0)
    assert receipt['sampling'] == second['sampling']
    assert all(torch.equal(a['actions'], b['actions']) for a,b in zip(before, after))
    assert not torch.equal(before[0]['targets'], after[0]['targets'])


@pytest.mark.parametrize('fault', ['cursor', 'repeat', 'runtime', 'missing', 'wrong_policy', 'nonfinite'])
def test_failure_never_advances_or_retries(monkeypatch, fault):
    c, _, slots, config, calls, _ = setup(monkeypatch)
    c.collect(slots, policy_config=config, completed_updates=0)
    update = 1
    if fault == 'cursor': update = 2
    elif fault == 'repeat': update = 0
    elif fault == 'runtime': config = dict(config, runtime_profile='other')
    else:
        original = c.engine.rollout
        def broken(*args):
            result = original(*args)
            if fault == 'missing': result['records'].pop()
            elif fault == 'wrong_policy': result['adapter_id'] = 'foreign'
            else: result['records'][0]['candidate_logprobs'][0][0] = float('nan')
            return result
        c.engine.rollout = broken
    with pytest.raises(ValueError): c.collect(slots, policy_config=config, completed_updates=update)
    assert c.last_update == 0 and c.phase == 'failed'
    with pytest.raises(ValueError): c.collect(slots, policy_config=config, completed_updates=1)


def test_reference_configuration_is_immutable_and_resume_cursor_is_explicit(monkeypatch):
    c, _, slots, config, calls, _ = setup(monkeypatch)
    updated = copy.deepcopy(config)
    config['adapter_id'] = 'mutated_external_config'
    c.collect(slots, policy_config=updated, completed_updates=317)
    assert c.reference_id == 'base:reference' and calls == ['reference']

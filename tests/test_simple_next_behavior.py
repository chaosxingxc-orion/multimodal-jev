"""Next-update transport with real identity/sampling and mocked audio compiler I/O."""
import copy
import pytest
torch = pytest.importorskip('torch')
from multimodal_jev.simple_next_behavior import next_batch
from multimodal_jev.simple_audio_batch import compile_batch, renderer_for_policy
from test_simple_audio_batch import fixture


def setup(monkeypatch):
    population, slots, processor, calls = fixture(monkeypatch)
    slots[-1] = dict(slots[0], padding=True, weight=0.)
    _, renderer, _ = compile_batch(population, {}, slots, processor,
        policy_id='base:updated', global_batch=128)
    def result(rows, adapter, peak):
        p = torch.full((60,), .5/59); p[peak] = .5
        return dict(passed=True, backend='vllm', base_id='base', adapter_id=adapter,
            records=[dict(request_id=r[4], request_identity=r[5], candidate_logprobs=[p.log().tolist()]) for r in rows])
    behavior = result(renderer, 'updated', 7)
    reference = result(renderer_for_policy(renderer, 'base:stage1'), 'stage1', 2)
    kw = dict(behavior_result=behavior, reference_result=reference, policy_id='base:updated',
        reference_policy_id='base:stage1', completed_updates=1, seed=317)
    return population, slots, processor, kw


def test_new_behavior_fixed_reference_padding_and_label_independent_sampling(monkeypatch):
    population, slots, processor, kw = setup(monkeypatch)
    items, receipt = next_batch(population, {}, slots, processor, **kw)
    assert receipt['next_optimizer_update'] == 2
    assert receipt['sampling']['global_real_requests'] == 127
    assert receipt['sampling']['update'] == 1
    assert len({r['request_id'] for r in items}) == 128
    assert all(r['behavior'][0].argmax().item() == 7 for r in items)
    assert all(r['reference_policy'][0].argmax().item() == 2 for r in items)
    for row in population.values(): row['target'] = (row['target']+1)%60
    changed, again = next_batch(population, {}, slots, processor, **kw)
    assert receipt == again
    assert all(torch.equal(a['actions'], b['actions']) for a,b in zip(items, changed))
    assert not torch.equal(items[0]['targets'], changed[0]['targets'])
    assert items[-1]['request_weight'] == 0 and not items[-1]['real_request']


@pytest.mark.parametrize('fault', ['old_behavior', 'moving_reference', 'missing', 'duplicate',
    'foreign_input', 'zero_cursor', 'bool_cursor', 'same_policy', 'hf'])
def test_reject_stale_behavior_or_wrong_next_batch(monkeypatch, fault):
    population, slots, processor, kw = setup(monkeypatch)
    if fault == 'old_behavior': kw['behavior_result'] = copy.deepcopy(kw['reference_result'])
    elif fault == 'moving_reference': kw['reference_result'] = copy.deepcopy(kw['behavior_result'])
    elif fault == 'missing': kw['behavior_result']['records'].pop()
    elif fault == 'duplicate': kw['behavior_result']['records'].append(kw['behavior_result']['records'][0])
    elif fault == 'foreign_input': kw['behavior_result']['records'][0]['request_identity']['request']['audio'] = 'other'
    elif fault == 'zero_cursor': kw['completed_updates'] = 0
    elif fault == 'bool_cursor': kw['completed_updates'] = True
    elif fault == 'same_policy': kw['reference_policy_id'] = kw['policy_id']
    else: kw['behavior_result']['backend'] = 'hf'
    with pytest.raises(ValueError): next_batch(population, {}, slots, processor, **kw)

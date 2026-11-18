"""Actual tensor identities and G16 replay, with substituted audio/engine I/O."""
import copy
import json
import pytest

pytest.importorskip('torch')
from multimodal_jev import batched_stage2_recovery_policy as m
from multimodal_jev.baseline import file_hash
from test_batched_stage2_training_audit import fixture as training_fixture


def fixture(tmp_path, monkeypatch):
    _, _, root, put = training_fixture(tmp_path, monkeypatch)
    config = json.loads((root / 'config.json').read_text())
    config['continuous_resume'] = dict(training_root=str(root))
    rollout = json.loads((root / 'run/rollout-after-0002.json').read_text())
    recovered = dict(completed_updates=2, policy_id=rollout['base_id'] + ':' + rollout['adapter_id'],
        reference_policy_id='base:reference', checkpoint_id='accepted-checkpoint',
        files={str(p): file_hash(p) for p in root.rglob('*') if p.is_file()})
    from transformers import Qwen3ASRProcessor
    processor = Qwen3ASRProcessor.from_pretrained('base')
    prepared = m.prepare(config, recovered, processor)
    reports = [dict(rank=rank, binding=copy.deepcopy(prepared['binding']), modes=[
        dict(microbatch=micro, records=copy.deepcopy(prepared['original'][rank]))
        for micro in (1, 8)]) for rank in range(4)]
    return config, recovered, processor, prepared, reports, root, put


def test_recompile_sampling_and_both_restored_modes(tmp_path, monkeypatch):
    _, _, _, prepared, reports, _, _ = fixture(tmp_path, monkeypatch)
    result = m.verify_reports(reports, prepared)
    assert result['passed'] and result['updates'] == 2 and result['comparisons'] == 256
    assert [r['microbatch'] for r in result['checks']] == [1, 8]
    assert all(r['maximum_logprob_error'] == 0 for r in result['checks'])
    assert len(prepared['items']) == 128


@pytest.mark.parametrize('fault', ['missing_rank', 'missing_slot', 'duplicate_slot', 'rank_order',
    'cursor', 'policy', 'checkpoint', 'micro', 'profile', 'permutation', 'probability', 'nan',
    'missing_single', 'endpoint_probability'])
def test_false_pass_cannot_admit_restored_policy(tmp_path, monkeypatch, fault):
    _, _, _, prepared, reports, _, _ = fixture(tmp_path, monkeypatch)
    report = reports[0]
    report['passed'] = True
    if fault == 'missing_rank': reports.pop()
    elif fault == 'missing_slot': report['modes'][0]['records'].pop()
    elif fault == 'duplicate_slot': report['modes'][0]['records'][1] = report['modes'][0]['records'][0]
    elif fault == 'rank_order': report['modes'][0]['records'].reverse()
    elif fault == 'cursor': report['binding']['updates'] += 1
    elif fault == 'policy': report['binding']['policy_id'] = 'base:foreign'
    elif fault == 'checkpoint': report['binding']['checkpoint_id'] = 'foreign'
    elif fault == 'micro': report['modes'][1]['microbatch'] = 16
    elif fault == 'profile': report['binding']['candidate_profile'] = 'foreign'
    elif fault == 'permutation': report['binding']['candidate_rank_indices'][0].reverse()
    elif fault in ('probability', 'nan'):
        report['modes'][1]['records'][0]['candidate_logprobs'][0][0] += .1 if fault == 'probability' else float('nan')
    elif fault == 'endpoint_probability':
        prepared['original'][0][0]['candidate_logprobs'][0][0] += .1
    else: report['modes'].pop(0)
    with pytest.raises(ValueError): m.verify_reports(reports, prepared)


@pytest.mark.parametrize('fault', ['changed_bytes', 'unbound', 'sampling', 'permutation', 'wrong_endpoint'])
def test_replay_requires_authenticated_and_recomputed_inputs(tmp_path, monkeypatch, fault):
    config, recovered, processor, _, _, root, put = fixture(tmp_path, monkeypatch)
    path = root / 'run/inputs-0001.json'
    data = json.loads(path.read_text())
    if fault == 'unbound': recovered['files'].pop(str(path))
    elif fault == 'wrong_endpoint': recovered['policy_id'] = 'base:foreign'
    else:
        if fault in ('changed_bytes', 'sampling'): data['sampling']['sampling'][0]['actions'][0][0][0] ^= 1
        else: data['candidate_rank_indices'][0].reverse()
        put(path, data)
        if fault != 'changed_bytes': recovered['files'][str(path)] = file_hash(path)
    with pytest.raises(ValueError): m.prepare(config, recovered, processor)


@pytest.mark.parametrize('corrupt', [False, True])
def test_runtime_gate_executes_no_grad_single_and_micro_before_return(tmp_path, monkeypatch, corrupt):
    import torch
    import torch.distributed as dist
    import multimodal_jev.span_audio_collate as collate
    config, recovered, processor, prepared, reports, _, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(m, 'prepare', lambda *a: prepared)
    def gather(peers, local):
        if isinstance(local, dict) and 'modes' in local:
            peers[:] = [local, *reports[1:]]
        else:
            peers[:] = [local] * 4
    monkeypatch.setattr(dist, 'all_gather_object', gather)
    calls = []
    by_id = {r['request_id']: r for r in prepared['original'][0]}
    def forward(model, group, **kwargs):
        assert not torch.is_grad_enabled()
        calls.append(len(group))
        result = [(torch.tensor(by_id[r['request_id']]['candidate_logprobs']),) for r in group]
        if corrupt: result[0][0][0, 0] += .1
        return result
    monkeypatch.setattr(collate, 'forward_span_audio_batch', forward)
    root = tmp_path / 'resumed'
    (root / 'rank-0').mkdir(parents=True)
    def check():
        return m.check_restored_policy(None, processor, config, recovered, root,
                                      rank=0, device='cpu', collation={})
    if corrupt:
        with pytest.raises(ValueError): check()
        assert not (root / 'rank-0/restored-policy-verified.json').exists()
    else:
        assert check()['passed']
        saved = json.loads((root / 'rank-0/restored-policy.json').read_text())
        assert saved == reports[0]
    assert calls == [1] * 32 + [8] * 4


def test_zero_weight_tail_keeps_all_slots_in_recovery_comparison(monkeypatch):
    from test_simple_resident_batch import setup
    from multimodal_jev.simple_audio_batch import renderer_for_policy
    from multimodal_jev.span_batch_training_config import PROFILE, batch_layout, reorder_global_items
    collector, _, slots, config, _, _ = setup(monkeypatch)
    slots[65:] = [dict(slots[0], padding=True, weight=0.) for _ in range(63)]
    items, receipt = collector.collect(slots, policy_config=config, completed_updates=316)
    current = dict(config, adapter_id='updated', population=dict(requests=65))
    rollout = collector.engine.rollout(current, renderer_for_policy(collector.last_renderer, 'base:updated'))
    rollout['reference_compared'] = False
    layout_config = dict(global_batch=128, microbatch_per_rank=8, candidate_profile=PROFILE, batch_order='length_grouped')
    ordered, indices = reorder_global_items(items, layout_config)
    logp = rollout['records'][0]['candidate_logprobs']
    original = [[dict(request_id=items[i]['request_id'], candidate_logprobs=copy.deepcopy(logp))
                 for i in rank] for rank in indices]
    binding = dict(updates=317, policy_id='base:updated', checkpoint_id='tail', batch_layout=batch_layout(layout_config))
    prepared = dict(binding=binding, items=ordered, original=original,
        sampling=receipt['sampling'], rollout=rollout, rank_ids=[[r['request_id'] for r in rows] for rows in original])
    reports = [dict(rank=r, binding=copy.deepcopy(binding), modes=[
        dict(microbatch=micro, records=copy.deepcopy(original[r])) for micro in (1, 8)]) for r in range(4)]
    result = m.verify_reports(reports, prepared)
    assert result['comparisons'] == 256 and result['checks'][0]['unique_inputs'] == 65
    assert sum(i['real_request'] for i in ordered) == 65
    padding_rank, padding_index = next((r, j) for r in range(4) for j in range(32)
                                       if not ordered[r * 32 + j]['real_request'])
    reports[padding_rank]['modes'][1]['records'][padding_index]['candidate_logprobs'][0][0] += .1
    with pytest.raises(ValueError): m.verify_reports(reports, prepared)

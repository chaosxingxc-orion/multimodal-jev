import json

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_macro_rollout import digest
from tools.reports import context_transfer_macro_parity, run_native_seal_endpoint as endpoint


@pytest.fixture
def stage1():
    initial = {'step': 0, 'policy_id': 'initial'}
    trained = {'step': 2, 'policy_id': 'trained'}
    return {
        'schema': 'jev-infonce-source-evaluation-plan-v1',
        'status': 'audited_source_policies_bound_evaluation_pending',
        'algorithm': 'candidate_infonce',
        'checkpoints': [initial, trained],
        'reference_endpoint': dict(trained),
        'experiment': {'task_stage': {'updates': 2}},
        'source_audit': {'status': 'formal_infonce_artifacts_audited',
                         'updates': [{'step': 2, 'checkpoint': True}]},
    }


def test_hf_admits_stage1_terminal_and_bound_stage2_terminal(stage1):
    point = stage1['checkpoints'][-1]
    cache = {'dataset': 'MINDS_native14', 'seal': {'sha256': 'seal'}}
    endpoint.validate_hf_endpoint(stage1, point, cache)
    heldout = {'schema': 'jev-stage2-view-heldout-endpoint-v1', 'checkpoints': [point],
               'fixed_terminal_step': 2, 'evaluation_dataset': cache['dataset'], 'seal_sha256': 'seal'}
    endpoint.validate_hf_endpoint(heldout, point, cache)
    with pytest.raises(ValueError, match='held-out seal'):
        endpoint.validate_hf_endpoint(heldout, point, dict(cache, seal={'sha256': 'different'}))


@pytest.mark.parametrize('failure', ['initial', 'policy', 'audit', 'source_selection'])
def test_hf_rejects_other_stage1_points_or_inputs(stage1, failure):
    point = stage1['checkpoints'][-1]
    cache = {'dataset': 'MINDS_native14'}
    if failure == 'initial':
        point = stage1['checkpoints'][0]
    elif failure == 'policy':
        stage1['reference_endpoint']['policy_id'] = 'different'
    elif failure == 'audit':
        stage1['source_audit']['updates'][-1]['checkpoint'] = False
    else:
        cache['dataset'] = 'SLURP_clean_SELECTION60'
    with pytest.raises(ValueError, match='audited Stage1|held-out dataset'):
        endpoint.validate_hf_endpoint(stage1, point, cache)


def test_stage1_hf_shards_require_exact_finite_reference_before_full(stage1, tmp_path, monkeypatch):
    assets = [{'id': str(i)} for i in range(4)]
    candidates = ['intent']
    cache = {'dataset': 'MINDS_native14', 'panel_size': 4}
    point = stage1['checkpoints'][-1]

    def encode(asset):
        return None, dict(asset, positions=[1], inputs={'input_ids': 'hash-'+asset['id']})

    def reference(selected, delta=0):
        records = [dict(id=a['id'], logits=[float(delta)], prob=[.5], request_identity={
            'policy_id': point['policy_id'], 'positions': r['positions'], 'inputs': r['inputs'],
        }) for a, r in zip(selected['assets'], selected['input_receipts'], strict=True)]
        return dict(schema='jev-target-width-hf-reference-v1', panel_sha256=digest(selected),
                    policy_id=point['policy_id'], records=records)

    def write(path, value):
        path.write_text(json.dumps(value))
        return {'path': path, 'sha256': file_hash(path)}

    panel = endpoint.selected_panel({'cache': cache}, assets, candidates, encode)
    spec = dict(devices=[5, 7], reference_panel=write(tmp_path/'panel.json', panel),
                reference=write(tmp_path/'reference.json', reference(panel)))
    cfg = dict(point=point, cache=cache, physical_devices=[5], hf_shards=dict(spec, index=0))
    monkeypatch.setattr(context_transfer_macro_parity, 'hf_reference',
                        lambda config, admitted, selected, bundle, encode, out: reference(selected))
    monkeypatch.setattr(endpoint, 'commit_json', write)
    first = tmp_path/'hf-shard-check-0'
    first.mkdir()
    receipt = endpoint.hf_sharded(cfg, stage1, 'hf-shard-check', first, {}, assets, candidates, encode)
    assert receipt['device'] == 5
    write(first/'terminal.json', {'exit_code': 0})
    full = dict(cfg, cache={'dataset': cache['dataset']})
    with pytest.raises(FileNotFoundError, match='hf-shard-check-1/terminal.json'):
        endpoint.hf_sharded(full, stage1, 'hf-shard-full', tmp_path/'full', {}, assets, candidates, encode)

    second = tmp_path/'hf-shard-check-1'
    second.mkdir()
    other = dict(cfg, physical_devices=[7], hf_shards=dict(spec, index=1))
    monkeypatch.setattr(context_transfer_macro_parity, 'hf_reference',
                        lambda config, admitted, selected, bundle, encode, out: reference(selected, delta=1e-9))
    with pytest.raises(ValueError, match='differs from the original HF reference'):
        endpoint.hf_sharded(other, stage1, 'hf-shard-check', second, {}, assets, candidates, encode)

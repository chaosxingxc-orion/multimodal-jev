import json

import pytest
import torch

from test_context_selection_curve import config as experiment_config, points
from test_context_macro_evaluation import setup
from multimodal_jev.baseline import file_hash
from tools.reports import evaluate_context_transfer_macro as entry
from tools.reports.audit_context_macro_evaluation import audit
from tools.reports import audit_context_transfer_macro as collector


def write(path, value):
    path.write_text(json.dumps(value))
    return dict(path=str(path), sha256=file_hash(path))


def source_config(tmp_path):
    scores = points()
    scores[2]['result']['metrics']['query_macro_set_f1'] = .8
    source = dict(schema='jev-macro-transfer-source-evaluation-v1',
        algorithm='dual_credit_grpo_floor_clipped_v1', experiment=experiment_config(),
        initial_policy_id='p0', selected_checkpoint=dict(step=128, policy_id='p128'),
        target_tuning_allowed=False, points=[])
    for point in scores:
        source['points'].append(dict(step=point['step'],
            result=write(tmp_path/f"score-{point['step']}.json", point['result'])))
    return dict(source_evaluation=write(tmp_path/'source.json', source),
                policy_role='common_Stage1', policy_id='p0')


@pytest.mark.parametrize('role,identity,step', [
    ('common_Stage1', 'p0', 0), ('source_selected_new_GRPO', 'p128', 128),
    ('fixed_endpoint_new_GRPO', 'p634', 634)])
def test_three_predeclared_roles_bound_to_recomputed_source_curve(tmp_path, role, identity, step):
    config = source_config(tmp_path)
    config.update(policy_role=role, policy_id=identity)
    binding = entry.source_binding(config)
    assert binding['step'] == step and binding['policy_id'] == identity


@pytest.mark.parametrize('fault', ['truncated', 'reorder', 'chosen', 'test_role', 'threshold',
    'tuning', 'stale_hash', 'changed_score', 'policy', 'unknown_role', 'algorithm'])
def test_bad_source_refused_before_opening_target_inputs(tmp_path, monkeypatch, fault):
    config = source_config(tmp_path)
    source = json.loads((tmp_path/'source.json').read_text())
    if fault == 'truncated': source['points'].pop()
    if fault == 'reorder': source['points'].reverse()
    if fault == 'chosen': source['selected_checkpoint'] = dict(step=64, policy_id='p64')
    if fault == 'threshold': source['experiment']['evaluation']['set_threshold'] = .4
    if fault == 'tuning': source['target_tuning_allowed'] = True
    if fault == 'algorithm': source['algorithm'] = 'grpo_v1'
    if fault in ('test_role', 'changed_score'):
        path = tmp_path/'score-128.json'
        value = json.loads(path.read_text())
        value['role'] = 'test'
        bound = write(path, value)
        if fault == 'test_role': source['points'][2]['result'] = bound
    if fault == 'policy': config['policy_id'] = 'other'
    if fault == 'unknown_role': config['policy_role'] = 'best_target'
    config['source_evaluation'] = write(tmp_path/'source.json', source)
    if fault == 'stale_hash': config['source_evaluation']['sha256'] = 'wrong'
    def forbidden(*args, **kwargs):
        pytest.fail('target inputs or engines must not be opened before source validation')
    monkeypatch.setattr(entry, 'load', forbidden)
    monkeypatch.setattr(entry, 'connect_pool', forbidden)
    with pytest.raises(ValueError): entry.run(config, {})


@pytest.mark.parametrize('dataset,count', [('FSC_native31', 31), ('MINDS_native14', 14), ('SLURP_clean_TEST60',60)])
def test_shared_pool_complete_target_with_independent_journal_audit(tmp_path, monkeypatch, dataset, count):
    config = source_config(tmp_path)
    config['source_release']=dict(path='fixture',sha256=experiment_config()['release_sha256'])
    engine, population, encode, args, clients = setup(tmp_path, monkeypatch, n=5)
    candidates = [f'c{i}' for i in range(count)]
    def expanded(asset):
        row, receipt = encode(asset)
        row['positions'] = torch.arange(count).reshape(1, count)
        receipt.update(candidate_ids=candidates, positions=list(range(count)))
        return row, receipt
    independent = [expanded(asset)[1] for asset in population]
    for client in clients:
        original = client.score_many
        def scored(batch, _original=original, **kwargs):
            records = _original(batch, **kwargs)
            for row in records: row.update(logits=[0.]*count, prob=[.5]*count)
            return records
        client.score_many = scored
    bundle = dict(policy_id='p0', contract=dict(request_version='jev-context-exclusive_event-v1',
        runtime_profile=entry.PROFILE, token=dict(registration_status='admitted', id=151706)))
    monkeypatch.setattr(entry, 'check_paths', lambda paths: None)
    monkeypatch.setattr(entry, 'read_execution_bundle', lambda *a, **kw: bundle)
    monkeypatch.setattr(entry, 'load', lambda c: dict(population=population, candidate_ids=candidates,
        processor_sha256='fixture', cache_root=str(tmp_path/'cache')))
    monkeypatch.setattr(entry, 'audited_encoder', lambda *a, **kw: expanded)
    monkeypatch.setattr(entry, 'connect_pool', lambda *a, **kw: engine)
    config.update(dataset=dataset, inputs={'fixture': 'label-free'}, pins={}, snapshot='fixture',
        contract=bundle['contract'], base_id='base', output=str(tmp_path/'transfer'),
        macro_client=dict(runtime_identity=args['runtime_identity']), audio_token_id=9, macro_queries=4)
    result = entry.run(config, {'fixture': True})
    assert result['recordings'] == 5 and result['decisions'] == count
    assert result['target_scored'] is False and all(c.phase == 'closed' for c in clients)
    checked = audit(tmp_path/'transfer/inference', engine.root, population=population,
        candidate_ids=candidates, input_receipts=independent, policy_id='p0',
        runtime_identity=args['runtime_identity'], macro_queries=4)
    assert checked['recordings'] == 5 and checked['batches'] == 2
    monkeypatch.setattr(collector, 'load', lambda c: dict(population=population,
        candidate_ids=candidates, input_receipts=independent, processor_sha256='fixture'))
    proof = collector.audit_outputs(config, tmp_path/'transfer', engine.root)
    assert proof['recordings'] == 5 and proof['target_scored'] is False
    result_path = tmp_path/'transfer/result.json'
    original = json.loads(result_path.read_text())
    for field, value in [('policy_role', 'best_target'), ('recordings', 4),
                         ('target_scored', True), ('predictions_sha256', 'wrong')]:
        write(result_path, dict(original, **{field: value}))
        with pytest.raises(ValueError, match='independent reconstruction'):
            collector.audit_outputs(config, tmp_path/'transfer', engine.root)
    write(result_path, original)
    source_path = tmp_path/'transfer/source-binding.json'
    saved = json.loads(source_path.read_text())
    saved['source_evaluation']['sha256'] = 'different_source'
    write(source_path, saved)
    with pytest.raises(ValueError, match='source role'):
        collector.audit_outputs(config, tmp_path/'transfer', engine.root)
    with pytest.raises(FileExistsError): entry.run(config, {})

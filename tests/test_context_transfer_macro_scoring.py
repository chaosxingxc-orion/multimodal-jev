import copy
import json

import pytest
import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_logits_wire import WIRE
from multimodal_jev.context_macro_rollout import digest
from tools.reports import score_context_transfer_macro as module


def fixture(tmp_path, monkeypatch, dataset, *, alias=False):
    width,count={'FSC_native31':(31,3793),'MINDS_native14':(14,8168),'SLURP_clean_TEST60':(60,12991)}[dataset]
    candidates = [f'c{i}' for i in range(width)]
    variants = ['en-US', 'en-GB', 'en-AU', 'de-DE', 'fr-FR', 'es-ES', 'it-IT',
                'pt-PT', 'ru-RU', 'zh-CN', 'ja-JP', 'ko-KR', 'nl-NL', 'pl-PL']
    population = [dict(id=str(i), audio_sha256=f'a{i}', speaker_id=f's{i%10}',source_group=f'g{i//3}') for i in range(count)]
    truth = {str(i): (candidates[i%width] if width !=14 else
             dict(intent=candidates[i%width], language=variants[i%14])) for i in range(count)}
    receipts = [dict(id=r['id'], audio_sha256=r['audio_sha256'], positions=[list(range(width))],
        tensors={'input_ids': {'sha256': r['id']}}) for r in population]
    admitted = dict(candidate_ids=candidates, cache_receipts=receipts)
    entries = {}; proofs = {}; calls = []
    def binding(config):
        expected = {'common_Stage1': 'p0', 'source_selected_new_GRPO': 'p0' if alias else 'p1',
                    'fixed_endpoint_new_GRPO': 'p2'}
        if config['policy_id'] != expected[config['policy_role']]: raise ValueError('policy role differs')
        return {'policy_id': config['policy_id']}
    for index, role in enumerate(module.ROLES):
        policy = 'p0' if alias and index == 1 else f'p{index}'
        directory = tmp_path/policy
        if not directory.exists():
            (directory/'inference').mkdir(parents=True)
            (directory/'source-binding.json').write_text(json.dumps({'role': role}))
            predictions = []
            for i, (row, receipt) in enumerate(zip(population, receipts, strict=True)):
                identity = dict(policy_id=policy, positions=receipt['positions'][0], inputs=receipt['tensors'])
                logits = torch.full((width,), -3.)
                # Initial is wrong everywhere; selected correct; endpoint correct on half.
                chosen = i%width if index == 1 or index == 2 and i%2 else (i+1)%width
                logits[chosen] = 3.
                predictions.append(dict(id=row['id'], policy_id=policy, request_identity=identity,
                    request_id=digest(identity), logits=logits.tolist(), prob=logits.sigmoid().tolist(),
                    output_protocol=WIRE))
            path = directory/'inference/predictions.json'
            path.write_text(json.dumps({'records': predictions}))
            proofs[str(directory)] = dict(policy_id=policy, predictions_sha256=file_hash(path))
        entries[role] = dict(evaluation=dict(dataset=dataset, policy_role=role, policy_id=policy,
            source_evaluation={'sha256': 'source'}, inputs={'frozen': True},
            source_release={'path':'release','sha256':'release'},
            macro_client={'runtime_identity': {'profile': 'fixture'}}),
            collection=str(directory), journal=str(directory/'journal'))
    def audit(config, root, journal):
        calls.append(config['policy_role']); binding(config)
        return copy.deepcopy(proofs[root])
    monkeypatch.setattr(module, 'source_binding', binding)
    monkeypatch.setattr(module, 'audit_outputs', audit)
    monkeypatch.setattr(module, 'load', lambda config: admitted)
    monkeypatch.setattr(module, 'fsc_truth', lambda *args: (population, truth))
    monkeypatch.setattr(module, 'minds_truth', lambda *args: (population, truth))
    from tools.reports import context_slurp_test_macro_truth
    monkeypatch.setattr(context_slurp_test_macro_truth,'load',lambda *args:(population,truth))
    return entries, calls


@pytest.mark.parametrize('dataset', ['FSC_native31', 'MINDS_native14','SLURP_clean_TEST60'])
def test_full_population_three_roles_keep_endpoint_regression_and_language_metrics(tmp_path, monkeypatch, dataset):
    entries, calls = fixture(tmp_path, monkeypatch, dataset)
    report = module.score_matrix(entries, {})
    assert report['unique_policies'] == 3 and len(calls) == 3
    assert not report['runtime_provenance_verified'] and not report['target_tuning']
    selected = report['policies']['source_selected_new_GRPO']
    endpoint = report['policies']['fixed_endpoint_new_GRPO']
    metrics = 'recording_weighted' if dataset == 'MINDS_native14' else 'metrics'
    assert report['evaluation_kind']==('source_domain_held_out' if dataset=='SLURP_clean_TEST60' else 'cross_dataset_transfer')
    assert selected[metrics]['top1_event_accuracy'] == 1
    assert .49 < endpoint[metrics]['top1_event_accuracy'] < .51
    if dataset == 'MINDS_native14':
        assert len(endpoint['variants']) == 14 and len(endpoint['languages']) == 12
        assert len(report['comparisons']['fixed_endpoint_new_GRPO']['variant_delta']) == 14


def test_identical_policy_reuses_real_predictions_preserving_both_roles(tmp_path, monkeypatch):
    entries, calls = fixture(tmp_path, monkeypatch, 'FSC_native31', alias=True)
    report = module.score_matrix(entries, {})
    assert report['unique_policies'] == 2 and len(calls) == 2
    assert report['policies']['common_Stage1'] == report['policies']['source_selected_new_GRPO']
    assert all(v == 0 for v in report['comparisons']['source_selected_new_GRPO']['delta'].values())


@pytest.mark.parametrize('fault', ['missing_role', 'role', 'runtime', 'source', 'inputs', 'failed_audit', 'changed_predictions'])
def test_incomplete_or_changed_matrix_cannot_open_target_gold(tmp_path, monkeypatch, fault):
    entries, calls = fixture(tmp_path, monkeypatch, 'FSC_native31')
    later = entries['fixed_endpoint_new_GRPO']
    if fault == 'missing_role': del entries['fixed_endpoint_new_GRPO']
    if fault == 'role': later['evaluation']['policy_role'] = 'source_selected_new_GRPO'
    if fault == 'runtime': later['evaluation']['macro_client']['runtime_identity'] = {'other': True}
    if fault == 'source': later['evaluation']['source_evaluation']['sha256'] = 'other'
    if fault == 'inputs': later['evaluation']['inputs'] = {'other': True}
    if fault == 'failed_audit':
        original = module.audit_outputs
        def audit(config, *args):
            if config['policy_role'] == 'fixed_endpoint_new_GRPO': raise ValueError('failed audit')
            return original(config, *args)
        monkeypatch.setattr(module, 'audit_outputs', audit)
    if fault == 'changed_predictions':
        from pathlib import Path
        (Path(later['collection'])/'inference/predictions.json').write_text('{}')
    def forbidden(*args): pytest.fail('gold opened before complete matrix admission')
    monkeypatch.setattr(module, 'fsc_truth', forbidden)
    monkeypatch.setattr(module, 'minds_truth', forbidden)
    with pytest.raises(ValueError): module.score_matrix(entries, {})

import copy
import importlib.util
import json
from pathlib import Path

import pytest

import tools.jev.audit_context_rl_curve as curve


def setup(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('plan_fixture', Path(__file__).with_name('test_context_rl_evaluation_plan.py'))
    mod = importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    args = mod.fixture(tmp_path)
    config = args['experiment']
    evaluations, reports = [], {}
    for step in config['evaluation']['checkpoints']:
        evaluations.append(dict(step=step, collection=tmp_path / f'eval-{step}',
            transport=dict(container_id=f'container-{step}'), expected_config_sha256=f'config-{step}'))
        metrics = {k: .5 for k in [config['evaluation']['primary'], *config['evaluation']['secondary']]}
        metrics.update(queries=9226, threshold=.5, candidate_ids=[str(i) for i in range(60)],
                       binary_nll_infinite=False, set_precision_defined_queries=9226)
        reports[step] = dict(passed=True, step=step, run_id='test', algorithm='grpo_v1',
            source_policy_id='initial' if step == 0 else f'rl-{step}', initial_policy_id='initial',
            inference_policy_id=f'export-{step}', container_id=f'container-{step}',
            result=dict(policy_id=f'export-{step}', role='selection',
                release_sha256=config['release_sha256'], predictions_sha256=f'pred-{step}', metrics=metrics))
    calls = []
    def audit(**kwargs):
        assert Path(kwargs['stage1_root']) == Path(args['stage1_root'])
        assert Path(kwargs['rl_root']) == Path(args['rl_root'])
        calls.append(kwargs['step'])
        return copy.deepcopy(reports[kwargs['step']])
    monkeypatch.setattr(curve, 'audit', audit)
    return dict(evaluations=evaluations, **args), reports, calls


def test_reaudits_all_points_preserves_both_identities(tmp_path, monkeypatch):
    args, reports, calls = setup(tmp_path, monkeypatch)
    reports[128]['result']['metrics']['query_macro_set_f1'] = .8
    report = curve.run(**args)
    assert calls == args['experiment']['evaluation']['checkpoints']
    assert report['complete'] and report['runtime_provenance_verified']
    assert report['initial_policy_id'] == 'export-0'
    assert report['initial_source_policy_id'] == 'initial'
    assert report['selected_checkpoint'] == dict(step=128, policy_id='export-128')
    assert report['selected_source_checkpoint'] == dict(step=128, policy_id='rl-128')
    assert report['primary_adjacent_decreases'][0]['step'] == 192
    assert reports[0]['result']['policy_id'] == 'export-0'


def test_prefix_audits_but_never_selects(tmp_path, monkeypatch):
    args, _, calls = setup(tmp_path, monkeypatch);args['evaluations'] = args['evaluations'][:3]
    report = curve.run(**args)
    assert calls == [0, 64, 128]
    assert not report['complete'] and report['selected_checkpoint'] is None
    assert report['selected_source_checkpoint'] is None


@pytest.mark.parametrize('fault', ['missing', 'reorder', 'override', 'duplicate_root', 'duplicate_handle',
    'wrong_source', 'wrong_run', 'wrong_algorithm', 'wrong_initial', 'relabel', 'failed', 'wrong_population'])
def test_curve_rejects_mixed_or_invalid_points(tmp_path, monkeypatch, fault):
    args, reports, calls = setup(tmp_path, monkeypatch);points = args['evaluations'];r = reports[64]
    if fault == 'missing':del points[1]
    elif fault == 'reorder':points[0], points[1] = points[1], points[0]
    elif fault == 'override':points[1]['algorithm'] = 'rloo_v1'
    elif fault == 'duplicate_root':points[1]['collection'] = points[0]['collection']
    elif fault == 'duplicate_handle':points[1]['transport'] = points[0]['transport']
    elif fault == 'wrong_source':r['source_policy_id'] = 'rl-128'
    elif fault == 'wrong_run':r['run_id'] = 'other'
    elif fault == 'wrong_algorithm':r['algorithm'] = 'rloo_v1'
    elif fault == 'wrong_initial':r['initial_policy_id'] = 'other'
    elif fault == 'relabel':r['result']['policy_id'] = 'rl-64'
    elif fault == 'failed':r['passed'] = False
    elif fault == 'wrong_population':r['result']['metrics']['queries'] = 9225
    with pytest.raises(ValueError):curve.run(**args)


def test_point_audit_failure_aborts_curve(tmp_path, monkeypatch):
    args, _, _ = setup(tmp_path, monkeypatch)
    def fail(**kwargs):raise ValueError('actual weights differ')
    monkeypatch.setattr(curve, 'audit', fail)
    with pytest.raises(ValueError, match='actual weights differ'):curve.run(**args)


def test_cli_spec_binds_input_files(tmp_path, monkeypatch):
    args, _, calls = setup(tmp_path, monkeypatch)
    inputs = tmp_path / 'inputs';inputs.mkdir()
    for name in ('experiment', 'stage1_proof', 'rl_proof'):
        path = inputs / (name + '.json');path.write_text(json.dumps(args[name]));args[name] = str(path)
    args['stage1_root'] = str(args['stage1_root']);args['rl_root'] = str(args['rl_root'])
    for p in args['evaluations']:
        path = inputs / f"transport-{p['step']}.json";path.write_text(json.dumps(p['transport']))
        p['transport'] = str(path);p['collection'] = str(p['collection'])
    spec = inputs / 'spec.json';spec.write_text(json.dumps(args))
    report = curve.from_spec(spec)
    assert report['complete'] and len(calls) == 11
    assert report['input_artifacts']['spec']['sha256'] == curve.sha(spec)
    assert len(report['input_artifacts']) == 15

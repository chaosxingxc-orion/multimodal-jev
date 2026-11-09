import json
from pathlib import Path
import pytest
from multimodal_jev.training_budget import batch_budget
from multimodal_jev.trajectory_budget import audit_trajectory


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def trajectory(root):
    stages = []
    for i in range(2):
        paths = {role: str(root/str(i)/role) for role in ('rollout', 'actor', 'export')}
        for path in paths.values():
            write(Path(path)/'terminal.json', {'exit_code': 0})
        write(Path(paths['rollout'])/'result.json', {
            'status': 'passed', 'prompts': 1, 'rollouts': 2, 'policy_id': str(i),
            'engine_load_seconds': 3., 'generation_budget': {
                'requests': 1, 'responses': 2, 'request_prompt_tokens': 10,
                'replicated_prompt_tokens': 20, 'generated_tokens': 5,
                'prompt_tokens_per_request': [10], 'response_tokens_per_record': [2, 3],
                'length_terminated_responses': 0, 'engine_generate_seconds': 2.}})
        write(Path(paths['actor'])/'result.json', {
            'status': 'passed', 'updates': 1,
            'config': {'rollout_run': paths['rollout'],
                       'parent_run': stages[-1]['actor'] if stages else None},
            'metrics': {'updated': i == 0, 'training_budget': batch_budget(
                [10, 10], [2, 3], [90, 90], gradient_passes=int(i == 0),
                no_grad_passes=1, optimizer_steps=int(i == 0))}})
        write(Path(paths['export'])/'result.json', {'policy_id': str(i+1), 'export_seconds': 4.})
        stages.append(paths)
    return stages


def change(stages, i, role, fn):
    path = Path(stages[i][role])/'result.json'
    data = json.loads(path.read_text())
    fn(data)
    write(path, data)


def test_update_then_noop_accounts_work_without_summing_cumulative_steps(tmp_path):
    report = audit_trajectory(trajectory(tmp_path))
    assert report['all_stages_completed'] and report['accounting_complete']
    cost = report['costs']
    assert cost['training']['total']['optimizer_steps'] == 1
    assert cost['training']['total']['gradient_tokens'] == 25
    assert cost['training']['total']['no_grad_tokens'] == 50
    assert cost['generation']['total']['generated_tokens'] == 10
    assert cost['engine_load_seconds']['total'] == 6
    assert cost['export_seconds']['total'] == 8
    assert report['iterations'][0]['evidence']['actor']['sha256']['result.json']


def test_legacy_missing_metrics_are_not_zero(tmp_path):
    stages = trajectory(tmp_path)
    change(stages, 0, 'actor', lambda data: data['metrics'].pop('training_budget'))
    change(stages, 1, 'rollout', lambda data: data.pop('generation_budget'))
    change(stages, 1, 'export', lambda data: data.pop('export_seconds'))
    report = audit_trajectory(stages)
    assert report['all_stages_completed'] and not report['accounting_complete']
    cost = report['costs']['training']
    assert cost['total'] is None and cost['observed_iterations'] == 1
    assert cost['observed_subtotal']['optimizer_steps'] == 0
    assert report['costs']['generation']['total'] is None


@pytest.mark.parametrize('terminal,state', [(None, 'unconfirmed'), ({'exit_code': 1}, 'failed')])
def test_residual_result_is_not_counted_without_success(tmp_path, terminal, state):
    stages = trajectory(tmp_path)
    path = Path(stages[1]['actor'])/'terminal.json'
    if terminal is None:
        path.unlink()
    else:
        write(path, terminal)
    report = audit_trajectory(stages)
    assert report['iterations'][1]['evidence']['actor']['state'] == state
    assert report['costs']['training']['total'] is None
    assert not report['all_stages_completed']


@pytest.mark.parametrize('role,mutate,match', [
    ('actor', lambda d: d.update(updates=2), 'cumulative'),
    ('actor', lambda d: d['config'].update(parent_run='/wrong'), 'parent'),
    ('actor', lambda d: d['config'].update(rollout_run='/wrong'), 'different rollout'),
    ('actor', lambda d: d['metrics']['training_budget'].update(optimizer_steps=1), 'update outcome'),
    ('actor', lambda d: d['metrics']['training_budget'].update(sequences=3), 'sequence'),
    ('rollout', lambda d: d.update(policy_id='stale'), 'preceding exported'),
    ('rollout', lambda d: d['generation_budget'].update(generated_tokens=7), 'receipt counts'),
    ('rollout', lambda d: d.update(engine_load_seconds=float('nan')), 'finite'),
])
def test_inconsistent_evidence_rejected(tmp_path, role, mutate, match):
    stages = trajectory(tmp_path)
    change(stages, 1, role, mutate)
    with pytest.raises(ValueError, match=match):
        audit_trajectory(stages)


def test_duplicate_stages_rejected(tmp_path):
    stages = trajectory(tmp_path)
    with pytest.raises(ValueError, match='double-count'):
        audit_trajectory([stages[0], stages[0]])


def test_success_requires_result(tmp_path):
    stages = trajectory(tmp_path)
    (Path(stages[0]['export'])/'result.json').unlink()
    with pytest.raises(ValueError, match='without result'):
        audit_trajectory(stages)

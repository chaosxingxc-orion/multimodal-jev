#!/usr/bin/env python3
"""Execute preregistered τ² TRAIN oracle/control traces; never a model score."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from multimodal_jev.baseline import atomic_json
from multimodal_jev.tau2_bridge import Tau2EnvironmentBridge


def run(output):
    from tau2.domains.retail.environment import get_environment, get_tasks, get_tasks_split

    splits = get_tasks_split()
    assert splits['train'][0] == '0'
    task = next(task for task in get_tasks('train') if task.id == '0')
    actions = task.evaluation_criteria.actions
    assert len(actions) == 5 and actions[-1].name == 'exchange_delivered_order_items'
    assert task.initial_state is None
    full = Tau2EnvironmentBridge(get_environment, task)
    initial = full.state_hashes()
    public = full.observation()
    assert set(public) == {'policy', 'tools', 'messages'} and public['messages'] == []
    # A sentinel mutation of private criteria must not affect the observation.
    private_task = task.model_copy(deep=True)
    private_task.evaluation_criteria.actions[0].arguments['first_name'] = 'PRIVATE_CRITERION_SENTINEL'
    assert Tau2EnvironmentBridge(get_environment, private_task).observation() == public
    responses = []
    for action in actions:
        response = full.step(call_id=action.action_id, name=action.name, arguments=action.arguments)
        assert not response.error and response.execution_status == 'executed'
        responses.append(response.model_dump(mode='json'))
    final = full.state_hashes()
    assert final != initial and full.replay_hashes() == final
    score = full.environment_score()
    assert score.db_check.db_match and score.db_check.db_reward == 1.
    assert score.reward == 1.
    # Duplicate IDs must fail before any additional tool execution.
    try:
        full.step(call_id=actions[-1].action_id, name=actions[-1].name, arguments=actions[-1].arguments)
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate call accepted')
    assert full.state_hashes() == final
    negative = Tau2EnvironmentBridge(get_environment, task)
    assert negative.state_hashes() == initial
    for action in actions[:-1]:
        response = negative.step(call_id=action.action_id, name=action.name, arguments=action.arguments)
        assert not response.error
    negative_score = negative.environment_score()
    assert negative.state_hashes() == initial
    assert not negative_score.db_check.db_match and negative_score.reward == 0.
    corrupt = full.trajectory()
    corrupt[-1].content = '{"corrupted_write_result":true}'
    try:
        full.replay_hashes(corrupt)
    except (ValueError, AssertionError) as exc:
        rejection = {'type': type(exc).__name__, 'message': str(exc)}
    else:
        raise AssertionError('strict replay accepted corrupt write response')
    # Full trajectories remain in shared evidence, outside Git.
    atomic_json(output/'oracle-trajectory.json', [m.model_dump(mode='json') for m in full.trajectory()])
    atomic_json(output/'omitted-write-trajectory.json', [m.model_dump(mode='json') for m in negative.trajectory()])
    data = Path(os.environ['TAU2_DATA_DIR'])/'tau2/domains/retail'
    hashes = {name: hashlib.sha256((data/name).read_bytes()).hexdigest()
              for name in ['tasks.json', 'split_tasks.json', 'db.json', 'policy.md']}
    return {'status': 'passed', 'domain': 'retail', 'task_id': task.id, 'split': 'train',
            'actions': len(actions), 'initial_hashes': initial, 'final_hashes': final,
            'oracle_environment_score': score.model_dump(mode='json'),
            'omitted_write_environment_score': negative_score.model_dump(mode='json'),
            'strict_replay_corruption_rejection': rejection,
            'private_criteria_observation_invariance': True, 'duplicate_call_rejected': True,
            'data_sha256': hashes,
            'scope': 'oracle TRAIN environment interface validation only; environment component, '
                     'not NL assertions, model predictions, speech interaction, policy compliance or academic reproduction'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        atomic_json(args.output/'result.json', run(args.output))
        atomic_json(args.output/'terminal.json', {'exit_code': 0, 'finished_unix': time.time()})
    except BaseException as exc:
        atomic_json(args.output/'terminal.json', {'exit_code': 1, 'error': repr(exc), 'finished_unix': time.time()})
        raise


if __name__ == '__main__':
    main()

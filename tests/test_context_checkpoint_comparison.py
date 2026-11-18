import json
from pathlib import Path

import pytest
import torch

from tools.jev.context_checkpoint_comparison import compare


@pytest.fixture
def pair(tmp_path):
    left, right = tmp_path / 'left', tmp_path / 'right'
    for root, interval in ((left, 1), (right, 64)):
        (root / 'run').mkdir(parents=True)
        (root / 'config.json').write_text(json.dumps(dict(start_update=0, stop_update=2,
            engineering_prefix=True, training_dispatch_allowed=False, checkpoint_interval=interval,
            algorithm='grpo_v1', pins={str(root): 'pin'}, schedule=str(root / 'schedule'),
            reference_cache=str(root / 'cache'), source_policy_id='same')))
        for step in (1, 2):
            (root / f'run/batch-{step-1:04d}.json').write_text(json.dumps({'id':step, 'policy_id':str(step-1)}))
            torch.save([dict(actions=torch.tensor([0, 1]), behavior=torch.tensor([.2, .8]))],
                       root / f'run/groups-{step-1:04d}.pt')
            (root / f'run/policy-{step:04d}').mkdir()
            torch.save({'w':torch.tensor([step], dtype=torch.bfloat16)}, root / f'run/policy-{step:04d}/execution.pt')
        (root / 'run/checkpoint-0002-committed.json').write_text(json.dumps(dict(updates=2, contract={'source':'fixed'})))
        torch.save({'w':torch.ones(2)}, root / 'run/execution-0002.pt')
        for rank in range(4):
            (root / f'run/rank-{rank}').mkdir()
            torch.save(dict(parameters={'w':torch.ones(2)}, optimizer={'step':torch.tensor(2),
                'exp_avg':torch.tensor([.1]), 'exp_avg_sq':torch.tensor([.01])}),
                root / f'run/rank-{rank}/checkpoint-0002.pt')
    (left / 'run/checkpoint-0001-committed.json').write_text('{}')
    return left, right


def test_same_training_different_save_frequency(pair):
    result = compare(*pair)
    assert result['passed'] and result['exact_final_masters_and_adam']
    assert 'not fresh-process replay' in result['scope']


@pytest.mark.parametrize('fault', ['adam', 'first_policy', 'sample', 'algorithm', 'source', 'extra_checkpoint', 'cursor'])
def test_reject_unmatched_runs(pair, fault):
    left, right = pair
    if fault == 'adam':
        p = right / 'run/rank-3/checkpoint-0002.pt';d = torch.load(p, weights_only=True)
        d['optimizer']['exp_avg_sq'] += .01;torch.save(d, p)
    elif fault == 'first_policy':
        torch.save({'w':torch.tensor([2], dtype=torch.bfloat16)}, right / 'run/policy-0001/execution.pt')
    elif fault == 'sample':
        p = right / 'run/groups-0000.pt';d = torch.load(p, weights_only=True)
        d[0]['actions'][0] = 1;torch.save(d, p)
    elif fault in ('algorithm', 'source', 'cursor'):
        p = right / 'config.json';d = json.loads(p.read_text())
        d[{'algorithm':'algorithm', 'source':'source_policy_id', 'cursor':'start_update'}[fault]] = 'changed'
        p.write_text(json.dumps(d))
    elif fault == 'extra_checkpoint':
        (right / 'run/checkpoint-0001-committed.json').write_text('{}')
    with pytest.raises(ValueError):compare(left, right)

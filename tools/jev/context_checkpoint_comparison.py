"""Compare independently collected interval1/64 two-update runs, not a replay."""
import json
from pathlib import Path

import torch

from tools.jev.context_replay_audit import require_exact


def compare(left, right):
    left, right = Path(left), Path(right)
    def read(root, name):return json.loads((root / name).read_text())
    def tensor(root, name):return torch.load(root / name, map_location='cpu', weights_only=True)
    a, b = read(left, 'config.json'), read(right, 'config.json')
    if (a['checkpoint_interval'] != 1 or b['checkpoint_interval'] != 64
            or any(c['start_update'] != 0 or c['stop_update'] != 2
                   or c['engineering_prefix'] is not True or c['training_dispatch_allowed'] is not False
                   for c in (a, b))):
        raise ValueError('bounded matched interval1/64 runs required')
    # Paths and source pins differ by design; every other setting must match.
    excluded = {'pins', 'schedule', 'reference_cache', 'checkpoint_interval'}
    require_exact({k:v for k,v in a.items() if k not in excluded},
                  {k:v for k,v in b.items() if k not in excluded}, 'config')
    for step in (1, 2):
        require_exact(read(left, f'run/batch-{step-1:04d}.json'),
                      read(right, f'run/batch-{step-1:04d}.json'), 'batch')
        require_exact(tensor(left, f'run/groups-{step-1:04d}.pt'),
                      tensor(right, f'run/groups-{step-1:04d}.pt'), 'groups')
        require_exact(tensor(left, f'run/policy-{step:04d}/execution.pt'),
                      tensor(right, f'run/policy-{step:04d}/execution.pt'), 'policy')
    ac, bc = read(left, 'run/checkpoint-0002-committed.json'), read(right, 'run/checkpoint-0002-committed.json')
    require_exact(ac['contract'], bc['contract'], 'contract')
    if ac['updates'] != 2 or bc['updates'] != 2:
        raise ValueError('checkpoint cursor differs')
    require_exact(tensor(left, 'run/execution-0002.pt'), tensor(right, 'run/execution-0002.pt'), 'execution')
    for rank in range(4):
        name = f'run/rank-{rank}/checkpoint-0002.pt'
        require_exact(tensor(left, name), tensor(right, name), f'rank-{rank}')
    if (right / 'run/checkpoint-0001-committed.json').exists():
        raise ValueError('interval run unexpectedly saved step1')
    return dict(passed=True, updates=2, exact_samples=True, exact_per_step_execution=True,
                exact_final_masters_and_adam=True, rank_shards=4,
                scope='Matched checkpoint-frequency comparison only; not fresh-process replay or efficacy')

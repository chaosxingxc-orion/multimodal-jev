"""Bounded CPU check of native verl on authenticated archived reward groups."""
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch
from verl.trainer.ppo import core_algos


def run(report):
    expected = 'fadfafa83d94ba40d7ac2abeedb6e99ea0716b42f8744c421ecbb371db8b4d24'
    source = Path(inspect.getfile(core_algos))
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError('Pinned verl source differs')
    rows = report['queries']
    if len(rows) != 128 or len({r['request_id'] for r in rows}) != 128:
        raise ValueError('Complete unique archived global128 required')
    rewards = torch.tensor([r['rewards'] for r in rows], dtype=torch.float32)
    if rewards.shape != (128, 16):
        raise ValueError('G16 required')
    flat = rewards.reshape(-1, 1)
    mask = torch.ones_like(flat)
    index = np.array([r['request_id'] for r in rows for _ in range(16)], dtype=object)
    centered = rewards - rewards.mean(-1, keepdim=True)
    formulas = dict(grpo=centered / (rewards.std(-1, correction=1, keepdim=True) + 1e-6),
                    rloo=centered * (16 / 15))
    grpo, _ = core_algos.compute_grpo_outcome_advantage(flat.clone(), mask, index, norm_adv_by_std_in_grpo=True)
    rloo, _ = core_algos.compute_rloo_outcome_advantage(flat.clone(), mask, index)
    result = {}
    constant = (rewards == rewards[:, :1]).all(-1)
    for name, value in [('grpo', grpo), ('rloo', rloo)]:
        value = value.reshape_as(rewards)
        error = float((value - formulas[name]).abs().max())
        if not torch.isfinite(value).all() or error > 2e-6:
            raise ValueError(f'{name} formula/native mismatch: {error}')
        result[name] = dict(max_formula_error=error,
                            nonzero_advantage_groups=int((value.abs().sum(-1) > 0).sum()),
                            constant_group_max_abs=float(value[constant].abs().max()),
                            advantages=value.tolist())
    return dict(core_algos_path=str(source), core_algos_sha256=actual,
                torch_version=torch.__version__, algorithms=result,
                scope='Native CPU advantage estimators on authenticated archived rewards; not NPU actor/gradient replay or efficacy.')


if __name__ == '__main__':
    torch.set_num_threads(1)
    p = Path(sys.argv[1])
    result = run(json.loads(p.read_text()))
    result['input_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    result['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print('JEV_NATIVE_REPLAY_RESULT=' + json.dumps(result), flush=True)

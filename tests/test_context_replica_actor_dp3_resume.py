"""Real native verl/DDP DP3 updates at local64, including the one-real tail.

Synthetic FP32 logits; this does not certify audio/BF16 FSDP or NPU capacity.
"""
import json
from pathlib import Path

import pytest
import torch.multiprocessing as mp

from test_context_dual_actor_dp_resume import worker


@pytest.mark.parametrize('real_rows', [192, 1])
def test_dp3_local64_grpo_native_accumulation_and_resume(tmp_path, real_rows):
    pytest.importorskip('verl'); pytest.importorskip('omegaconf')
    assert not list(Path('/dev').glob('davinci[0-9]*'))
    algorithm = 'dual_credit_grpo_floor_clipped_v1'
    settings = (3, 192, real_rows, (16, 32), (algorithm,))
    mp.spawn(worker, args=(str(tmp_path), False, *settings), nprocs=3, join=True)
    mp.spawn(worker, args=(str(tmp_path), True, *settings), nprocs=3, join=True)
    for rank in range(3):
        continuous = json.loads((tmp_path/f'continuous-rank{rank}.json').read_text())
        resumed = json.loads((tmp_path/f'replay-rank{rank}.json').read_text())
        assert continuous['passed'] and len(continuous['cases']) == 4
        assert resumed['passed'] and len(resumed['cases']) == 2
        assert {c['micro'] for c in resumed['cases']} == {16, 32}
        assert all(c['replay_exact'] and c['algorithm'] == algorithm for c in resumed['cases'])

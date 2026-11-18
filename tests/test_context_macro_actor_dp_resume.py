"""DP2/global128 native verl CPU updates and exact fresh-process recovery.

Uses synthetic logits, real Gloo/DDP and the production dual-loss adapter.
This does not establish BF16 FSDP/audio/NPU parity or throughput.
"""
import json
from pathlib import Path

import pytest
import torch.multiprocessing as mp

from test_context_dual_actor_dp_resume import worker


def test_dp2_global128_grpo_micro8_16_32_and_fresh_process_resume(tmp_path):
    pytest.importorskip('verl'); pytest.importorskip('omegaconf')
    assert not list(Path('/dev').glob('davinci[0-9]*'))
    algorithm = 'dual_credit_grpo_floor_clipped_v1'
    settings = (2, 128, 65, (8, 16, 32), (algorithm,))
    mp.spawn(worker, args=(str(tmp_path), False, *settings), nprocs=2, join=True)
    mp.spawn(worker, args=(str(tmp_path), True, *settings), nprocs=2, join=True)
    for rank in range(2):
        continuous = json.loads((tmp_path/f'continuous-rank{rank}.json').read_text())
        resumed = json.loads((tmp_path/f'replay-rank{rank}.json').read_text())
        assert continuous['passed'] and len(continuous['cases']) == 6
        assert resumed['passed'] and len(resumed['cases']) == 3
        assert {c['micro'] for c in resumed['cases']} == {8, 16, 32}
        assert all(c['replay_exact'] and c['algorithm'] == algorithm for c in resumed['cases'])

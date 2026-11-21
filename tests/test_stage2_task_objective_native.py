"""Fixed-image native verl acceptance; no surrogate replacement."""
import hashlib
from pathlib import Path

import torch

from extensions.stage2_task_objective import DEFAULTS
from multimodal_jev.objective_plugin import ObjectivePlugin
from tests.test_objective_plugin import batch


def test_real_pinned_verl_joint_objective_is_finite_and_padding_safe():
    p=Path(__file__).resolve().parents[1]/'extensions/stage2_task_objective.py'
    plugin=ObjectivePlugin(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),options=DEFAULTS))
    args=batch();loss,metrics=plugin.hybrid_loss(*args,global_real_requests=2)
    gradient,=torch.autograd.grad(loss,args[0])
    assert torch.isfinite(gradient).all() and not gradient[2].any()
    assert metrics['sampled_set_count']==32

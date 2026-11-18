import pytest
import torch
from tools.reports.context_probability_policy_v2 import binary_logprobs
from tools.reports.prepare_context_arithmetic_candidate import transform_actor, transform_trainer
from pathlib import Path
from tools.reports.audit_context_arithmetic_candidate import validate_config


def test_actual_probability_logs_and_full_backward():
    torch.manual_seed(317)
    layer=torch.nn.Linear(3,4)
    x=torch.randn(8,3)
    z=layer(x)
    weights=torch.randn(8,4,2)
    logs=torch.stack(binary_logprobs(z),1)
    prob=z.detach().sigmoid()
    assert torch.equal(logs.detach(),torch.stack((torch.log1p(-prob),torch.log(prob)),-1))
    loss=(logs*weights).sum()
    got=torch.autograd.grad(loss,tuple(layer.parameters()))
    analytic=weights[...,1]*(1-prob)-weights[...,0]*prob
    torch.testing.assert_close(got[0],analytic.T@x,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(got[1],analytic.sum(0),atol=2e-6,rtol=2e-6)


@pytest.mark.parametrize('bad',[torch.zeros(3),torch.zeros(0,2),torch.ones(2,3,dtype=torch.int64),
                              torch.zeros(2,3,dtype=torch.float64),torch.tensor([[100.]]),
                              torch.tensor([[-1000.]]),torch.tensor([[float('nan')]])])
def test_invalid_or_saturated_distribution_rejected(bad):
    with pytest.raises(ValueError):binary_logprobs(bad)


def test_bf16_uses_actual_fp32_sampler():
    z=torch.tensor([[-6.,0.,3.84375]],dtype=torch.bfloat16,requires_grad=True)
    logs=torch.stack(binary_logprobs(z),1)
    assert logs.dtype==torch.float32
    p=z.detach().float().sigmoid()
    assert torch.equal(logs.detach(),torch.stack((torch.log1p(-p),torch.log(p)),-1))
    logs.sum().backward()
    assert torch.isfinite(z.grad).all()


def test_candidate_changes_only_declared_import_and_checkpoint_identity():
    p=Path('src/multimodal_jev/context_native_actor.py');text=p.read_text()
    changed=transform_actor(text)
    assert changed.replace('from context_probability_policy_v2 import binary_logprobs',
                           'from .context_binary_policy import binary_logprobs')==text
    trainer=Path('.local/decision-objective-research/context-checkpoint-shared-npu-v3-package/source/tools/jev/train_context_online_candidate.py').read_text()
    new=transform_trainer(trainer)
    assert "contract['binary_policy_arithmetic'] = ARITHMETIC" in new
    assert "stop_update-start_update>2" in new
    assert 'save_now=((index+1)' in new
    with pytest.raises(ValueError):transform_trainer(new)
    with pytest.raises(ValueError):transform_actor(changed)


def test_arithmetic_admission_rejects_formal_or_relaxed_bounds():
    from tools.reports.context_probability_policy_v2 import PROFILE
    c=dict(stage='task',engineering_prefix=True,training_dispatch_allowed=False,
        stop_update=2,start_update=0,global_batch=128,microbatch=8,physical_devices=[4,5,6,7],
        runtime_profile='qwen3-asr-context-bf16-packed-row-v1',binary_policy_arithmetic=PROFILE,
        algorithm='grpo_v1',checkpoint_interval=1,
        behavior_bounds=dict(probability_max_abs=2e-7,binary_logprob_max_abs=2e-6))
    validate_config(c)
    for change in [dict(stop_update=634),dict(training_dispatch_allowed=True),dict(microbatch=32),
                   dict(binary_policy_arithmetic='old'),dict(algorithm='ppo'),
                   dict(behavior_bounds=dict(probability_max_abs=2e-7,binary_logprob_max_abs=1e-5))]:
        with pytest.raises(ValueError):validate_config({**c,**change})

"""CPU arithmetic in the pinned verl image. No model or accelerator required."""
import json
import math
from pathlib import Path

import torch

from multimodal_jev import context_dual_credit as dual
from multimodal_jev.grpo_numerics import current, use
from multimodal_jev.objective_plugin import ObjectivePlugin

ROOT=Path(__file__).resolve().parents[1]


def spec(floor,clip):
    return dict(advantage_std_floor=floor,policy_clip=dict(clip_ratio=clip,
                clip_ratio_high=None,clip_ratio_low=None,clip_ratio_c=3.0))


def test_native_verl_clipping_separates_behavior_from_current():
    # The authentic actor normally verifies on-policy parity. Exercise clipping
    # directly with a deliberately different behavior probability.
    for epsilon,expected in ((.1,0.),(.5,-1.3)):
        new=torch.tensor([[math.log(.65)]],requires_grad=True)
        with use(spec(.25,epsilon)):
            loss=dual.native_surrogate(dict(new_logprob=new,old_logprob=torch.tensor([[math.log(.5)]]),
                advantages=torch.ones(1,1),weight=torch.ones(1)))
        torch.testing.assert_close(torch.autograd.grad(loss,new)[0],torch.tensor([[expected]]))
    assert current() is None


def test_real_pinned_objective_uses_configured_floor_and_preserves_defaults():
    design=json.loads((ROOT/'configs/experiments/stage2-without-set-credit-v1.json').read_bytes())
    plugin=dict(path=str(ROOT/design['objective_source']),sha256=design['objective_sha256'],options=design['options'])
    def gradient(numerics):
        z=torch.tensor([[.7,-.4,1.2]],requires_grad=True)
        ident=dict(policy_id='p0',input_id='i',query_id='q',update=0)
        rec=dual.prepare(z[0].detach(),z[0].detach().sigmoid(),['a','b','c'],torch.tensor([True,False,False]),
                         **ident,algorithm=dual.ALGORITHMS[1])
        obj=ObjectivePlugin(dict(plugin,**({'numerics':numerics} if numerics is not None else {})))
        loss,_=obj.hybrid_loss(z,[rec],[ident],torch.full_like(z,.5).detach(),torch.tensor([True]),
                               global_real_requests=1)
        return loss.detach(),torch.autograd.grad(loss,z)[0]
    old,configured=gradient(None),gradient(spec(.25,.2))
    for a,b in zip(old,configured,strict=True):torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert not torch.allclose(configured[1],gradient(spec(1.,.2))[1])
    assert current() is None

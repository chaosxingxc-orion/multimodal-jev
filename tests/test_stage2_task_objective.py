import hashlib
from pathlib import Path

import pytest
import torch

from multimodal_jev.objective_plugin import ObjectivePlugin
from test_objective_plugin import batch, surrogate
from extensions.stage2_task_objective import DEFAULTS


def plugin(**overrides):
    p=Path(__file__).resolve().parents[1]/'extensions/stage2_task_objective.py'
    return ObjectivePlugin(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        options=dict(DEFAULTS,**overrides)))


def test_joint_rewards_score_exact_partial_wrong_and_empty():
    from types import SimpleNamespace
    from extensions.stage2_task_objective import compute_rewards
    actions=torch.tensor([[1,1,1,0,0],[0,1,1,1,0],[0,0,1,0,0]])
    branch=dict(actions=actions,target=torch.tensor([True,False,False]),rewards=actions.float())
    ctx=SimpleNamespace(options=DEFAULTS,records=[dict(branches={'existence':branch,'ranking':branch})])
    reward=compute_rewards(ctx)[0]['task'][0]
    torch.testing.assert_close(reward,torch.tensor([1.,1/3,1/4,0.,0.]))


def test_joint_plugin_padding_and_microbatch_equivalence(surrogate):
    args=batch(); objective=plugin()
    loss,metrics=objective.hybrid_loss(*args,global_real_requests=2)
    expected,=torch.autograd.grad(loss,args[0])
    assert not expected[2].any()
    assert metrics['sampled_set_count']==32
    gradients=[]
    for i in range(3):
        z=args[0].detach().clone().requires_grad_()
        part,_=objective.hybrid_loss(z[i:i+1],args[1][i:i+1],args[2][i:i+1],args[3][i:i+1],args[4][i:i+1],
                                    global_real_requests=2,world_size=3)
        gradients.append(torch.autograd.grad(part,z)[0])
    torch.testing.assert_close(sum(gradients)/3,expected)


def test_summed_logscore_corrects_saturated_false_positive(surrogate):
    from multimodal_jev import context_dual_credit as dual
    z=torch.tensor([[8.,7.,-7.]],requires_grad=True)
    identity=dict(policy_id='p',input_id='i',query_id='q',update=0)
    record=dual.prepare(z[0].detach(),z[0].detach().sigmoid(),['a','b','c'],
        torch.tensor([True,False,False]),**identity,algorithm=dual.ALGORITHMS[1])
    objective=plugin(ranking_rl_weight=0.,existence_rl_weight=0.,task_rl_weight=0.,
                     ranking_auxiliary_coefficient=0.,reference_kl_mean_coefficient=0.)
    loss,_=objective.hybrid_loss(z,[record],[identity],torch.full_like(z,.5).detach(),
                                torch.tensor([True]),global_real_requests=1)
    gradient,=torch.autograd.grad(loss,z)
    assert gradient[0,1]>.049 and gradient[0,0]<0


def test_joint_gradient_matches_exact_enumeration_for_tiny_event():
    # Verify product-policy score function against an independent enumerated
    # expected utility gradient before group standardization / clipping.
    import itertools
    from multimodal_jev.context_task_reward import score
    z=torch.tensor([.4,-.3],dtype=torch.float64,requires_grad=True);p=z.sigmoid()
    actions=torch.tensor(list(itertools.product((0,1),repeat=2)))
    labels=torch.tensor([[True,False]])
    rewards=score(actions[None],labels,torch.ones_like(labels),dict(schema='jev-task-set-reward-v1',
        semantics='exclusive_event',exact_weight=.5,overlap_weight=.5,beta=1.))['reward'][0].double()
    logp=torch.where(actions.bool(),p.log(),torch.log1p(-p)).sum(-1)
    expected=(logp.exp()*rewards).sum()
    score_function=(logp.exp().detach()*rewards*logp).sum()
    g1,=torch.autograd.grad(expected,z,retain_graph=True)
    g2,=torch.autograd.grad(score_function,z)
    torch.testing.assert_close(g1,g2)


def test_options_must_be_frozen_explicitly(surrogate):
    p=plugin();p.options.pop('task_rl_weight')
    with pytest.raises(ValueError,match='explicit complete'):
        p.hybrid_loss(*batch(),global_real_requests=2)


def test_task_plugin_rejects_incompatible_arm(surrogate):
    with pytest.raises(ValueError,match='joint arm'):
        plugin().hybrid_loss(*batch(),global_real_requests=2,arm='ranking_only')

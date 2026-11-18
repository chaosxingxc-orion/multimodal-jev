import hashlib
from pathlib import Path

import pytest
import torch

from multimodal_jev import context_dual_credit as dual
from multimodal_jev.objective_plugin import ObjectivePlugin
from tools.reports.context_plugin_trainer import generate,load_actor

ROOT=Path(__file__).resolve().parents[1]


def binding(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),options={})


@pytest.fixture
def surrogate(monkeypatch):
    # Independent CPU surrogate for host integration tests; native verl remains
    # a separate fixed-image acceptance requirement.
    def loss(t):
        new,old,adv=(t[k].float() for k in ('new_logprob','old_logprob','advantages'))
        ratio=(new-old).clamp(-20,20).exp()
        clipped=torch.maximum(-adv*ratio,-adv*ratio.clamp(.8,1.2))
        clipped=torch.where(adv<0,torch.minimum(-adv*3,clipped),clipped)
        return (clipped.mean(-1)*t['weight']).sum()
    monkeypatch.setattr(dual,'native_surrogate',loss)


def batch():
    z=torch.tensor([[.7,-.4,1.2],[-.2,.4,.8],[.1,-.3,.6]],requires_grad=True)
    labels=torch.eye(3,dtype=torch.bool)
    identities=[dict(policy_id='p0',input_id=f'i{i}',query_id=f'q{i}',update=0) for i in range(3)]
    records=[dual.prepare(z[i].detach(),z[i].detach().sigmoid(),['a','b','c'],labels[i],
        **identities[i],algorithm=dual.ALGORITHMS[1]) for i in range(3)]
    return z,records,identities,torch.full_like(z,.5).detach(),torch.tensor([True,True,False])


def test_default_exact_loss_gradient_and_padding(surrogate):
    args=batch();plugin=ObjectivePlugin(binding(ROOT/'extensions/stage2_objective.py'))
    old,_=dual.hybrid_loss(*args,global_real_requests=2)
    new,_=plugin.hybrid_loss(*args,global_real_requests=2)
    torch.testing.assert_close(new,old,rtol=0,atol=0)
    g0,=torch.autograd.grad(old,args[0],retain_graph=True)
    g1,=torch.autograd.grad(new,args[0])
    torch.testing.assert_close(g0,g1,rtol=0,atol=0)
    assert not g1[2].any()


def test_default_dp_microbatch_gradient_matches_full_batch(surrogate):
    args=batch();plugin=ObjectivePlugin(binding(ROOT/'extensions/stage2_objective.py'))
    full,_=plugin.hybrid_loss(*args,global_real_requests=2)
    expected,=torch.autograd.grad(full,args[0])
    gradients=[]
    for i in range(3):
        z=args[0].detach().clone().requires_grad_()
        loss,_=plugin.hybrid_loss(z[i:i+1],args[1][i:i+1],args[2][i:i+1],args[3][i:i+1],args[4][i:i+1],
                                 global_real_requests=2,world_size=3)
        gradients.append(torch.autograd.grad(loss,z)[0])
    torch.testing.assert_close(sum(gradients)/3,expected)


@pytest.mark.parametrize('change',['reward','loss'])
def test_editing_only_extension_changes_gradient(tmp_path,surrogate,change):
    text=(ROOT/'extensions/stage2_objective.py').read_text()
    if change=='reward':text=text.replace("branch['rewards'].clone()","1.0 - branch['rewards'].clone()")
    else:text=text.replace('    return ctx.baseline_loss(rewards)',
        '    loss, metrics = ctx.baseline_loss(rewards)\n'
        '    penalty = (ctx.logits.square().mean(-1) * ctx.real_requests.to(ctx.logits.device)).sum() * ctx.world_size / ctx.global_real_requests\n'
        '    return loss + 0.3 * penalty, metrics')
    p=tmp_path/'objective.py';p.write_text(text)
    args=batch();old,_=ObjectivePlugin(binding(ROOT/'extensions/stage2_objective.py')).hybrid_loss(*args,global_real_requests=2)
    new,_=ObjectivePlugin(binding(p)).hybrid_loss(*args,global_real_requests=2)
    g0,=torch.autograd.grad(old,args[0],retain_graph=True);g1,=torch.autograd.grad(new,args[0])
    assert not torch.allclose(g0,g1)
    assert not g1[2].any()


def test_hash_and_stale_behavior_rejected(tmp_path,surrogate):
    p=tmp_path/'objective.py';p.write_bytes((ROOT/'extensions/stage2_objective.py').read_bytes())
    c=binding(p);p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='file changed'):ObjectivePlugin(c)
    args=batch();args[2][0]['update']=1
    with pytest.raises(ValueError,match='stale'):ObjectivePlugin(binding(p)).hybrid_loss(*args,global_real_requests=2)


def test_invalid_reward_and_detached_loss_rejected(tmp_path,surrogate):
    base=(ROOT/'extensions/stage2_objective.py').read_text()
    p=tmp_path/'objective.py';p.write_text(base.replace("branch['rewards'].clone()","branch['rewards'].clone().requires_grad_()"))
    with pytest.raises(ValueError,match='detached'):ObjectivePlugin(binding(p)).hybrid_loss(*batch(),global_real_requests=2)
    p.write_text(base.replace('    return ctx.baseline_loss(rewards)',
        '    loss, metrics = ctx.baseline_loss(rewards)\n    return loss.detach(), metrics'))
    with pytest.raises(ValueError,match='differentiable'):ObjectivePlugin(binding(p)).hybrid_loss(*batch(),global_real_requests=2)


@pytest.mark.parametrize('formal',[False,True])
def test_real_generated_trainer_calls_extension_and_binds_checkpoint(formal):
    source=generate(ROOT,formal=formal)
    assert "load_dual_actor(c['source_root'],c['objective_plugin'])" in source
    assert "contract['objective_plugin']=dual_actor.objective_plugin_identity" in source
    compile(source,'<test>','exec')


def test_actor_scoped_hook_without_global_mutation():
    from tools.reports import context_dual_actor_candidate as actor
    old=actor.hybrid_loss;loaded=actor._loaded
    plugin=load_actor(ROOT,binding(ROOT/'extensions/stage2_objective.py'))
    assert plugin.objective_plugin_identity['sha256']==binding(ROOT/'extensions/stage2_objective.py')['sha256']
    assert actor.hybrid_loss is old and actor._loaded is loaded


@pytest.mark.parametrize('invalid',['shape','gradient','nan','boundary','arm','stale','parity'])
def test_custom_loss_cannot_skip_input_admission(tmp_path,invalid):
    # This plugin never calls baseline_loss or verl. Input checks must still run.
    p=tmp_path/'objective.py'
    p.write_text("API_VERSION = 'jev-stage2-objective-v1'\n"
        "def compute_rewards(ctx): return None\n"
        "def compute_loss(ctx, rewards):\n"
        "    value = (ctx.logits.square().mean(-1) * ctx.real_requests).sum()\n"
        "    return value * ctx.world_size / ctx.global_real_requests, dict(rl=0., auxiliary=0., kl=0.)\n")
    args=list(batch());kwargs=dict(global_real_requests=2)
    if invalid=='shape':args[3]=args[3][:1]
    elif invalid=='gradient':args[3].requires_grad_()
    elif invalid=='nan':args[3][0,0]=float('nan')
    elif invalid=='boundary':args[3][0,0]=1.
    elif invalid=='arm':kwargs['arm']='unknown'
    elif invalid=='stale':args[2][0]['update']=1
    elif invalid=='parity':args[0]=args[0]+.1
    with pytest.raises(ValueError):
        ObjectivePlugin(binding(p)).hybrid_loss(*args,**kwargs)


def test_actor_emits_custom_metrics_without_changing_loss(tmp_path,surrogate,monkeypatch):
    from types import SimpleNamespace
    from tools.reports import context_dual_actor_candidate as original
    monkeypatch.setattr(original.dist,'get_world_size',lambda:1)
    p=tmp_path/'objective.py'
    p.write_text((ROOT/'extensions/stage2_objective.py').read_text().replace(
        '    return ctx.baseline_loss(rewards)',
        '    loss, metrics = ctx.baseline_loss(rewards)\n'
        "    metrics.update(extra_brier=0.125, diagnostic='text', enabled=True)\n"
        '    return loss, metrics'))
    actor=load_actor(ROOT,binding(p));args=batch()
    expected,_=ObjectivePlugin(binding(p)).hybrid_loss(*args,global_real_requests=2)
    config=SimpleNamespace(policy_loss=SimpleNamespace(loss_mode=original.MODE),clip_ratio=.2,
                           ppo_mini_batch_size=24,ppo_micro_batch_size_per_gpu=3)
    payload=[dict(dual_credit=r,dual_identity=i,reference_probability=ref,
        real_request=bool(real),global_real_requests=2,dual_options=dict(arm='joint'))
        for r,i,ref,real in zip(*args[1:],strict=True)]
    logp=torch.zeros(3,6)
    token=actor._active.set(dict(consumed=False,log_probs=logp,logits=args[0],payload=payload))
    try:
        actual,metrics=actor.registered_loss(None,logp,None,None,config=config)
        torch.testing.assert_close(actual,expected*8,rtol=0,atol=0)
        assert metrics['jev/objective/extra_brier']==.125
        assert set(metrics)=={'jev/context_loss','jev/dual_rl','jev/dual_auxiliary',
                              'jev/dual_kl_unweighted','jev/objective/extra_brier'}
        with pytest.raises(ValueError,match='one exact dual forward'):
            actor.registered_loss(None,logp,None,None,config=config)
    finally:actor._active.reset(token)

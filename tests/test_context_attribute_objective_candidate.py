import json
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F
from tools.reports.build_context_attribute_candidate import catalog, targets
from tools.reports.context_attribute_objective_candidate import prepare_groups, loss
from tools.reports.context_probability_policy_v2 import binary_logprobs


def case(arm='q_credit'):
    # Use the actual 18/46/60 catalog shipped with the CPU verification package.
    path=Path('.local/decision-objective-research/slurp-native-attribute-prob-candidate-v1/catalog.json')
    m=json.loads(path.read_text()); chosen=[m['events'][i]['id'] for i in (0,17,59)]
    y=torch.tensor([targets(m,k)['labels'] for k in chosen],dtype=torch.bool)
    if arm=='direct':
        idx=[i for i,n in enumerate(m['nodes']) if n['family']=='final'];y=y[:,idx]
    mask=torch.ones_like(y);n=y.shape[1]
    z=torch.linspace(-1.2,1.1,3*n).reshape(3,n).requires_grad_()
    old=tuple(p.detach() for p in binary_logprobs(z.detach()*.8))
    ref=tuple(p.detach() for p in binary_logprobs(z.detach()*.6))
    generator=torch.Generator().manual_seed(81)
    a=torch.randint(0,2,(3,16,n),generator=generator)
    a[:,0,:]=y.long();a[:,1,:]=0
    # Last row is padding but has structurally valid native truth/actions.
    w=torch.tensor([1.,.5,0.]);real=torch.tensor([True,True,False])
    return m,y,mask,z,old,ref,a,w,real


def independent_loss(z,y,mask,a,old_joint,adv,ref,w,menu,arm,alpha=.3):
    p=z.sigmoid();logs=torch.stack((torch.log1p(-p),p.log()),-1)
    chosen=logs[:,None,:,:].expand(-1,16,-1,-1).gather(-1,a[...,None]).squeeze(-1).sum(-1)
    ratio=(chosen-old_joint).clamp(-20,20).exp()
    clipped=torch.maximum(-adv*ratio,-adv*ratio.clamp(.8,1.2))
    objective=torch.where(adv<0,torch.minimum(-adv*3,clipped),clipped)
    rl=(objective.mean(-1)*w).sum()/2
    reference=torch.stack(ref,1)
    kl=((logs.exp()*(logs-reference)).sum(-1).sum(-1)*w).sum()/2
    idx=list(range(z.shape[1])) if arm=='direct' else [i for i,n in enumerate(menu['nodes']) if n['family']=='final']
    aux=z.sum()*0
    for i in range(3):
        row=z[i,idx];truth=y[i,idx]
        rank=F.softplus(row[~truth][None,:]-row[truth][:,None]).mean()
        nll=F.binary_cross_entropy_with_logits(row,truth.float())
        aux=aux+w[i]*alpha*(rank+nll)/2
    return rl+.01*kl+aux


@pytest.mark.parametrize('algorithm',['grpo_v1','rloo_v1'])
@pytest.mark.parametrize('arm',['direct','q_input','q_credit'])
def test_native_joint_loss_gradient_and_microbatch_reduction(algorithm,arm):
    # Require actual verl: missing dependencies are errors, never skipped.
    m,y,mask,z,old,ref,a,w,real=case(arm)
    group=prepare_groups(old,a,y,mask,m,arm=arm,request_ids=['a','b','pad'],algorithm=algorithm)
    r=group['rewards'];centered=r-r.mean(-1,keepdim=True)
    expected_adv=centered/(r.std(-1,keepdim=True)+1e-6) if algorithm=='grpo_v1' else r-(r.sum(-1,keepdim=True)-r)/15
    torch.testing.assert_close(group['advantages'],expected_adv,rtol=1e-5,atol=1e-6)
    expected=independent_loss(z,y,mask,a,group['old_joint'],group['advantages'],ref,w,m,arm)
    expected_gradient=torch.autograd.grad(expected,z)[0]
    kwargs=dict(arm=arm,global_real_requests=2,ranking_coefficient=.3,probability_coefficient=.3)
    value,metrics=loss(z,y,mask,w,real,m,actions=a,old_joint=group['old_joint'],
        advantages=group['advantages'],reference_policy=ref,**kwargs)
    torch.testing.assert_close(value,expected,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(torch.autograd.grad(value,z)[0],expected_gradient,rtol=2e-4,atol=2e-6)
    assert metrics['sampled']['ce']==0 and torch.equal(expected_gradient[2],torch.zeros_like(z[2]))
    zz=z.detach().clone().requires_grad_();summed=zz.sum()*0
    for i in range(3):
        v,_=loss(zz[i:i+1],y[i:i+1],mask[i:i+1],w[i:i+1],real[i:i+1],m,
            actions=a[i:i+1],old_joint=group['old_joint'][i:i+1],advantages=group['advantages'][i:i+1],
            reference_policy=tuple(p[i:i+1] for p in ref),world_size=3,**kwargs)
        summed=summed+v/3
    torch.testing.assert_close(torch.autograd.grad(summed,zz)[0],expected_gradient,rtol=2e-4,atol=2e-6)


def test_q_arms_only_reward_changes_not_objective_given_same_advantage():
    m,y,mask,z,old,ref,a,w,real=case()
    group=prepare_groups(old,a,y,mask,m,arm='q_input',request_ids=['a','b','pad'],algorithm='grpo_v1')
    result=[];grads=[]
    for arm in ('q_input','q_credit'):
        zz=z.detach().clone().requires_grad_()
        v,_=loss(zz,y,mask,w,real,m,arm=arm,global_real_requests=2,ranking_coefficient=.3,
            probability_coefficient=.3,actions=a,old_joint=group['old_joint'],
            advantages=group['advantages'],reference_policy=ref)
        result.append(v.detach());grads.append(torch.autograd.grad(v,zz)[0])
    assert torch.equal(result[0],result[1]) and torch.equal(grads[0],grads[1])


def test_zero_signal_groups_keep_only_kl_and_final_auxiliary():
    m,y,mask,z,old,ref,a,w,real=case();a[:]=0
    group=prepare_groups(old,a,y,mask,m,arm='q_credit',request_ids=['a','b','pad'],algorithm='grpo_v1')
    assert torch.equal(group['advantages'],torch.zeros(3,16))
    v,_=loss(z,y,mask,w,real,m,arm='q_credit',global_real_requests=2,ranking_coefficient=.3,
        probability_coefficient=.3,actions=a,old_joint=group['old_joint'],
        advantages=group['advantages'],reference_policy=old)
    assert torch.isfinite(v) and torch.autograd.grad(v,z)[0][0].abs().sum()>0


@pytest.mark.parametrize('fault',['attached_behavior','wrong_support','G8','missing_truth','no_reference'])
def test_invalid_binding_rejected(fault):
    m,y,mask,z,old,ref,a,w,real=case()
    if fault=='attached_behavior':old=tuple(binary_logprobs(z))
    if fault=='wrong_support':old=old[:-1]
    if fault=='G8':a=a[:,:8]
    if fault=='missing_truth':mask[0,0]=False
    with pytest.raises(ValueError):
        group=prepare_groups(old,a,y,mask,m,arm='q_credit',request_ids=['a','b','pad'],algorithm='grpo_v1')
        if fault=='no_reference':
            loss(z,y,mask,w,real,m,arm='q_credit',global_real_requests=2,ranking_coefficient=.3,
                 probability_coefficient=.3,actions=a,old_joint=group['old_joint'],
                 advantages=group['advantages'],reference_policy=None)

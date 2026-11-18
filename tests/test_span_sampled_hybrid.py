import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.span_sampled_hybrid import loss
from multimodal_jev.span_sampled_hybrid import reference_kl_numerator


def test_exact_kl_is_once_per_request_and_reference_is_detached():
    logits=torch.tensor([[.2,.5],[-.2,.1]],requires_grad=True)
    policy=(logits.log_softmax(-1),)
    reference=(torch.tensor([[.6,.4],[.2,.8]]).log(),)
    weights=torch.tensor([1.5,0.])
    value=reference_kl_numerator(policy,reference,weights,.01)
    expected=1.5*sum(p*(p.log()-q.log()) for p,q in zip(policy[0][0].exp(),reference[0][0].exp()))
    torch.testing.assert_close(value,expected)
    gradient=torch.autograd.grad(value,logits)[0]
    assert gradient[0].abs().sum()>0 and torch.equal(gradient[1],torch.zeros(2))
    with pytest.raises(ValueError,match='fixed reference'):
        reference_kl_numerator(policy,None,weights,.01)
    with pytest.raises(ValueError):
        reference_kl_numerator(policy,(reference[0].requires_grad_(),),weights,.01)


def inputs():
    logits=torch.tensor([[.2,-.1,.4],[-.3,.6,.1],[.1,.2,-.1]],requires_grad=True)
    actions=torch.tensor([[[0],[1],[2],[0]],[[2],[1],[0],[1]],[[1],[0],[2],[2]]])
    old=torch.tensor([[.25,.25,.5],[.5,.3,.2],[.2,.3,.5]]).log().gather(1,actions[:,:,0])
    advantage=torch.tensor([[1.,-1.,.5,-.5],[0.,0.,0.,0.],[2.,-2.,1.,-1.]])
    return logits,actions,old,advantage,torch.tensor([[2],[1],[0]]),torch.ones(3,1,dtype=torch.bool),torch.ones(3,1),torch.tensor([.5,1.5,0.]),torch.tensor([True,True,False])


def test_native_sampled_hybrid_gradient_matches_explicit_full_loss_and_microbatches():
    pytest.importorskip('verl')
    z,a,old,adv,y,mask,sw,rw,real=inputs()
    p=z.log_softmax(-1);selected=p.gather(1,a[:,:,0]);ratio=(selected-old).clamp(-20,20).exp()
    clipped=torch.maximum(-adv*ratio,-adv*ratio.clamp(.8,1.2))
    dual=torch.where(adv<0,torch.minimum(-adv*3,clipped),clipped)
    expected=(dual.mean(-1)*rw).sum()/2+.4*(-p.gather(1,y)[:,0]*rw).sum()/2
    reference=torch.autograd.grad(expected,z)[0]
    actual,metrics=loss((z.log_softmax(-1),),a,old,adv,y,mask,sw,rw,real,global_real_requests=2,coefficient=.4)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual,z)[0],reference)
    # Simulate separate rank forwards followed by mean gradient reduction.
    gradients=[]
    for i in range(3):
        local=z.detach().clone().requires_grad_()
        value,_=loss((local[i:i+1].log_softmax(-1),),a[i:i+1],old[i:i+1],adv[i:i+1],
            y[i:i+1],mask[i:i+1],sw[i:i+1],rw[i:i+1],real[i:i+1],
            global_real_requests=2,coefficient=.4,world_size=3)
        gradients.append(torch.autograd.grad(value,local)[0])
    torch.testing.assert_close(sum(gradients)/3,reference)
    assert torch.equal(reference[2],torch.zeros(3))
    # Constant reward group still contributes CE and remains in denominator.
    assert reference[1].abs().sum()>0 and metrics['local_real_requests']==2


@pytest.mark.parametrize('fault',['padding','denominator','advantage_grad','coefficient','group'])
def test_invalid_metadata_rejected(fault):
    z,a,old,adv,y,mask,sw,rw,real=inputs();count=2;coefficient=.4
    if fault=='padding':rw[2]=1.
    if fault=='denominator':count=1
    if fault=='advantage_grad':adv.requires_grad_()
    if fault=='coefficient':coefficient=-1.
    if fault=='group':a=a[:,:1];old=old[:,:1];adv=adv[:,:1]
    with pytest.raises(ValueError):loss((z.log_softmax(-1),),a,old,adv,y,mask,sw,rw,real,
                                      global_real_requests=count,coefficient=coefficient)

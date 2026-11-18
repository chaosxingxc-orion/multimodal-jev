import copy
import itertools
import pytest
import torch
from tools.reports.build_context_attribute_candidate import catalog, targets
from tools.reports.context_attribute_reward_candidate import rewards, final_auxiliary


def menu():
    return catalog(dict(candidate_pairs={'a':['s1','x'],'b':['s1','y'],'c':['s2','x']},
        schema={'candidates':[dict(id=k,description=k) for k in 'abc']},thinking={'requirements':[]}))


def truth(m, event='a'):
    y=torch.tensor([targets(m,event)['labels']]);return y,torch.ones_like(y)


def test_exhaustive_actions_against_python_sets_and_matched_arms():
    m=menu();bits=list(itertools.product([0,1],repeat=7))
    actions=torch.tensor([bits]*3);y=torch.tensor([targets(m,e)['labels'] for e in 'abc']);mask=torch.ones_like(y)
    full,parts=rewards(actions,y,mask,m,arm='q_credit')
    plain,_=rewards(actions,y,mask,m,arm='q_input')
    direct,_=rewards(actions[:,:,4:],y[:,4:],mask[:,4:],m,arm='direct')
    assert torch.equal(plain,direct)
    for row in range(3):
        for j,b in enumerate(bits):
            expected={}
            for f,ids in [('scenario',[0,1]),('action',[2,3]),('final',[4,5,6])]:
                p={i for i in ids if b[i]};g={i for i in ids if y[row,i]}
                expected[f]=2*len(p&g)/(len(p)+len(g))
                assert float(parts[f][row,j])==pytest.approx(expected[f])
            assert float(full[row,j])==pytest.approx(expected['final']+.125*(expected['scenario']+expected['action']))
    assert not full.requires_grad and float(full.max())<=1.25


def test_no_attribute_final_repair_and_family_size_normalization():
    m=menu();y,mask=truth(m);actions=y.long()[:,None,:].repeat(1,2,1)
    actions[0,0,4:]=0
    score,parts=rewards(actions,y,mask,m,arm='q_credit')
    assert float(score[0,0])==.25 and float(parts['final'][0,0])==0
    assert float(score[0,1])==1.25
    actions[0,1,:4]=0
    assert float(rewards(actions,y,mask,m,arm='q_credit')[0][0,1])==1


def test_layout_permutation_preserves_reward():
    m=menu();y,mask=truth(m);g=torch.Generator().manual_seed(8)
    a=torch.randint(0,2,(1,16,7),generator=g);order=torch.randperm(7,generator=g)
    perm=copy.deepcopy(m);perm['nodes']=[m['nodes'][i] for i in order]
    torch.testing.assert_close(rewards(a,y,mask,m,arm='q_credit')[0],
        rewards(a[:,:,order],y[:,order],mask[:,order],perm,arm='q_credit')[0])


@pytest.mark.parametrize('fault',['missing','wrong_projection','no_final','bad_action','missing_member','duplicate_node'])
def test_bad_truth_or_catalog_rejected(fault):
    m=menu();y,mask=truth(m);a=y.long()[:,None,:].repeat(1,2,1)
    if fault=='missing':mask[0,0]=False
    if fault=='wrong_projection':y[0,:2]=y[0,:2].flip(0)
    if fault=='no_final':y[0,4:]=False
    if fault=='bad_action':a[0,0,0]=2
    if fault=='missing_member':m['nodes'][0]['members'].pop()
    if fault=='duplicate_node':m['nodes'][1]['id']=m['nodes'][0]['id']
    with pytest.raises(ValueError):rewards(a,y,mask,m,arm='q_credit')


def test_final_auxiliary_has_no_process_logit_gradient():
    m=menu();y,mask=truth(m);z=torch.linspace(-.8,.8,7,dtype=torch.float64)[None].requires_grad_()
    kw=dict(global_real_requests=1,ranking_coefficient=.3,probability_coefficient=.3)
    value,_=final_auxiliary(z,y,mask,torch.ones(1),torch.ones(1,dtype=torch.bool),m,arm='q_credit',**kw)
    grad=torch.autograd.grad(value,z)[0]
    assert torch.equal(grad[:,:4],torch.zeros_like(grad[:,:4])) and (grad[:,4:]!=0).all()
    direct,_=final_auxiliary(z[:,4:],y[:,4:],mask[:,4:],torch.ones(1),torch.ones(1,dtype=torch.bool),m,arm='direct',**kw)
    assert torch.equal(value,direct)
    other,_=final_auxiliary(z,y,mask,torch.ones(1),torch.ones(1,dtype=torch.bool),m,arm='q_input',**kw)
    assert torch.equal(value,other)

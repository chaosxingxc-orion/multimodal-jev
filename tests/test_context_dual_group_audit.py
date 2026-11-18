import copy

import pytest
import torch

from multimodal_jev.context_dual_credit import ALGORITHMS,prepare
from multimodal_jev.context_logits_wire import WIRE
from tools.reports.context_dual_group_audit import audit,digest


def fixture(algorithm=ALGORITHMS[0],labels=None):
    candidates=['b','a','c'];z=torch.tensor([.3,-.4,.7]);groups=[];receipts=[];targets={}
    for i in range(3):
        uid=f'q{i}';truth=labels if labels is not None else [i==j for j in range(3)]
        y=torch.tensor(truth,dtype=torch.bool);ident=dict(policy_id='p',inputs={'ids':str(i)},positions=[1,2,3])
        record=dict(policy_id='p',request_identity=ident,request_id=digest(ident),prob=z.sigmoid().tolist(),logits=z.tolist(),output_protocol=WIRE)
        refid=dict(ident,policy_id='ref');ref=dict(policy_id='ref',request_identity=refid,request_id=digest(refid),prob=[.5]*3)
        identity=dict(policy_id='p',input_id=record['request_id'],query_id=uid,update=0)
        credit=prepare(z,z.sigmoid(),candidates,y,**identity,algorithm=algorithm)
        opts=dict(algorithm=algorithm,arm='joint',ranking_coefficient=.1,probability_coefficient=.3,kl_coefficient=.01,reference_policy_id='ref')
        groups.append(dict(request_id=uid,policy_id='p',candidate_ids=candidates,labels=y[None],observed=torch.ones_like(y)[None],
            real_request=i!=2,request_weight=float(i!=2),global_rows=3,global_real_requests=2,dual_options=opts,
            reference_policy_id='ref',reference_probability=torch.tensor([.5]*3),dual_credit=credit,dual_identity=identity))
        receipts.append(dict(request_id=uid,behavior_record=record,reference_record=ref,credit_binding_sha256=credit['binding_sha256']))
        targets[uid]=dict(labels=truth,real_request=i!=2,weight=float(i!=2))
    return dict(groups=groups,receipts=receipts,targets=targets,candidates=candidates,
        algorithm=algorithm,arm='joint',update=0,policy_id='p',reference_policy_id='ref')


@pytest.mark.parametrize('algorithm',ALGORITHMS)
@pytest.mark.parametrize('labels',[None,[True,True,True],[False,False,False]])
def test_independent_rewards_samples_advantages_and_padding(algorithm,labels,monkeypatch):
    c=fixture(algorithm,labels)
    import multimodal_jev.context_dual_credit as production
    for name in ('prepare','verify','components','advantages'):
        monkeypatch.setattr(production,name,lambda *a,**kw:pytest.fail('production recomputation used'))
    result=audit(**c)
    assert result['checked_slots']==3 and result['real_queries']==2 and result['padding_slots']==1
    assert result['branches']['existence']['components']==6
    assert result['branches']['ranking']['components']==(4 if labels is None else 0)


@pytest.mark.parametrize('fault',['reward','advantage','action','weight','truth','reference','denominator','candidate','wire','duplicate'])
def test_independent_audit_rejects_changed_saved_evidence(fault):
    c=fixture();g=c['groups'][0];b=g['dual_credit']['branches']['ranking']
    if fault=='reward':b['rewards'][0,0]=1-b['rewards'][0,0]
    if fault=='advantage':b['advantages'][0,0]+=1
    if fault=='action':b['actions'][0,0]=1-b['actions'][0,0]
    if fault=='weight':b['weight'][0]=0
    if fault=='truth':c['targets']['q0']['labels']=[False,True,False]
    if fault=='reference':g['reference_probability'][0]=.6
    if fault=='denominator':g['global_real_requests']=3
    if fault=='candidate':g['candidate_ids']=list(reversed(g['candidate_ids']))
    if fault=='wire':c['receipts'][0]['behavior_record']['request_identity']['inputs']={'other':'value'}
    if fault=='duplicate':c['groups'][1]=copy.deepcopy(g)
    with pytest.raises(ValueError):audit(**c)


def test_frozen_fp32_arithmetic_remains_independently_auditable(monkeypatch):
    # An old artifact is explicitly labelled and reproduced by its frozen
    # numeric protocol; do not reinterpret it as a new FP64 trajectory.
    import multimodal_jev.context_dual_credit as production
    from dataclasses import replace
    original_components=production.components;original_binding=production._binding
    def old_components(z,ids,y,**kw):
        panel=original_components(z,ids,y,**kw);indices=sorted(range(len(ids)),key=lambda i:ids[i])
        pairs=[(i,j) for at,i in enumerate(indices) for j in indices[at+1:] if y[i]!=y[j]]
        q=torch.stack([z[i]-z[j] for i,j in pairs]).sigmoid() if pairs else z[:0]
        panel['ranking']=replace(panel['ranking'],probability=q)
        return panel
    def old_binding(*args,**kw):return dict(original_binding(*args,**kw),arithmetic='dual-credit-cpu-pair-fp32-v1')
    monkeypatch.setattr(production,'components',old_components);monkeypatch.setattr(production,'_binding',old_binding)
    c=fixture();assert audit(**c)['real_queries']==2
    for group,receipt in zip(c['groups'],c['receipts']):
        credit=group['dual_credit'];credit['binding']['arithmetic']='dual-credit-cpu-pair-fp64-v2'
        credit['binding_sha256']=digest(credit['binding']);receipt['credit_binding_sha256']=credit['binding_sha256']
    with pytest.raises(ValueError):audit(**c)

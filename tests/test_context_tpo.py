from collections import Counter
import pytest
import torch
from multimodal_jev.context_tpo import target, Objective, objectives
from multimodal_jev.tpo_view import schedule


def test_exact_distribution_and_frozen_gradient():
    ref = torch.tensor([[.2,.7,.4]], requires_grad=True)
    g = torch.tensor([0])
    u = torch.tensor([2.])
    q = target(ref,g,u,.5)
    weights = (ref.detach()/(1-ref.detach())) * torch.tensor([[torch.exp(torch.tensor(4.)),1.,1.]])
    torch.testing.assert_close(q.exp(), weights/weights.sum())
    z = torch.tensor([[.1,.4,.8]],requires_grad=True)
    cfg = Objective(.5,1,0,0,0,.1,0)
    loss, _ = objectives(z,q,g,g,torch.tensor([False]),cfg)
    loss.sum().backward()
    torch.testing.assert_close(z.grad,z.detach().softmax(-1)-q.exp())
    assert ref.grad is None
    perm=torch.tensor([2,0,1])
    torch.testing.assert_close(target(ref[:,perm],torch.tensor([1]),u,.5),q[:,perm])
    with pytest.raises(ValueError):target(torch.tensor([[0.,.9]]),g,u,.5)


def test_paired_online_gradient_and_padding():
    z=torch.tensor([[1.,3.],[3.,1.],[4.,6.]],requires_grad=True)
    g=torch.tensor([0,0,0])
    cfg=Objective(1,0,0,0,0,.1,1)
    loss, _=objectives(z,z.detach().log_softmax(-1),g,torch.tensor([1,0,2]),torch.tensor([1,1,0]),cfg)
    loss.sum().backward()
    assert z.grad[0,0]<0 and z.grad[1,0]>0 and not z.grad[2].any()


def test_schedule_preserves_population_and_pairs_through_tail():
    rows={str(i):dict(group=i//3 if i<8 else None,gold=0,audio_sha256=str(i)) for i in range(11)}
    batches=schedule(rows,seed=4,epochs=2,global_batch=8,microbatch=2)
    for epoch in (0,1):
        slots=[s for b in batches if b['epoch']==epoch for s in b['slots']]
        assert Counter(s['id'] for s in slots if not s['padding'])==Counter(rows.keys())
        for i,s in enumerate(slots):
            if s['paired']:
                p=slots[i^1]
                assert p['id']!=s['id'] and p['paired'] and rows[p['id']]['group']==rows[s['id']]['group']
            if s['padding']:assert s['weight']==0


def test_contrast_pairs_use_actual_competitor_and_push_same_candidate_apart():
    rows={'a':dict(group=1,gold=0,competitor=1,ref_gap=-.2,audio_sha256='a'),
          'b':dict(group=2,gold=1,competitor=0,ref_gap=-.1,audio_sha256='b')}
    slots=schedule(rows,seed=11,epochs=1,global_batch=2,microbatch=2,confusion_fraction=1.)[0]['slots']
    assert all(s['confusion'] and not s['paired'] for s in slots)
    z=torch.zeros((2,2),requires_grad=True)
    loss,_=objectives(z,z.detach().log_softmax(-1),torch.tensor([0,1]),torch.tensor([1,0]),
        torch.tensor([False,False]),Objective(1,0,0,0,0,.1,0,1),torch.tensor([True,True]))
    loss.mean().backward()
    assert z.grad[0,0]<0 and z.grad[1,0]>0 and z.grad[0,1]>0 and z.grad[1,1]<0


def test_full_population_increment_is_distinct_from_stage1_gain():
    from tools.reports.evaluate_tpo_endpoints import endpoint_deltas
    policies={'stage1':'one','direct':'two','tpo':'three'}
    data={ds:{arm:dict(policy_id=policy,queries=12,top1_macro_f1=f,top1_event_accuracy=f,
                      candidate_binary_nll=.2,candidate_brier=.1)
              for (arm,policy),f in zip(policies.items(),scores)}
          for ds,scores in [('source',[.7,.75,.85]),('transfer',[.6,.72,.65])]}
    population={ds:{'recordings':12} for ds in data}
    report=endpoint_deltas(data,population,policies,'source')
    assert report['observed_joint_gain_from_stage1']
    assert not report['observed_joint_increment_from_projection']
    data['source']['tpo']['queries']=11
    with pytest.raises(ValueError):endpoint_deltas(data,population,policies,'source')

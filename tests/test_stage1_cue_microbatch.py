import pytest
import torch
from multimodal_jev.stage1_cue_microbatch import compute

def test_semantic_binding_tail_gradient_and_split_rejection():
    names=['other','down','up']
    margin=dict(margin=.1,decision_threshold=.5,gap_weight=1.,anchor_weight=.25)
    native=dict(kind='candidate_infonce_probability_margin_v1',temperature=1.,probability_coefficient=0.,
                probability_margin=margin)
    z=torch.tensor([[.2,1.,-1.],[0.,-1.,1.],[.2,1.,-.2],[0.,-.2,1.],[1.,1.,1.]],requires_grad=True)
    y=torch.nn.functional.one_hot(torch.tensor([2,1,2,2,0]),3).bool()
    real=torch.tensor([True]*4+[False])
    records=[(str(i),str(i)*64) for i in range(5)]
    units={}
    for i,j,kind in [(0,1,'flip'),(2,3,'preserve')]:
        u=dict(kind=kind,family=['up','down'],rows=[dict(id=records[n][0],audio_sha256=records[n][1],
               gold=names[int(y[n].long().argmax())]) for n in (i,j)])
        units.update({records[n][0]:u for n in (i,j)})
    def run(idx,w,dp=1):
        return compute(z[idx],y[idx],torch.ones_like(y[idx]),real[idx].float(),real[idx],
            records=[records[i] for i in idx],units=units,candidates=names,global_real_requests=4,world_size=dp,
            objective=dict(native=native,flip_weight=w,preserve_weight=w))
    full,stats=run(list(range(5)),1)
    assert stats['flip_rows']==stats['preserve_rows']==2
    assert torch.allclose(full,(run([0,1],1,2)[0]+run([2,3,4],1,2)[0])/2)
    extra=torch.autograd.grad(full-run(list(range(5)),0)[0],z)[0]
    assert extra[0,2]<0 and extra[0,1]>0 and extra[1,2]>0 and extra[1,1]<0
    assert extra[2,2]<0 and extra[3,2]>0 and not extra[:,0].any() and not extra[4].any()
    with pytest.raises(ValueError,match='complete disjoint pair'):run([0],1)
    records[0]=('0','wrong')
    with pytest.raises(ValueError,match='binding'):run([0,1],1)

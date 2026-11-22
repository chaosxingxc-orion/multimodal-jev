import torch
from multimodal_jev.context_tpo_probability_gap import probability_gap
from tools.reports.build_tpo_gap_targets import shuffled_batches


def test_live_competitor_and_padding_gradient():
    z = torch.tensor([[0.,1.,2.],[0.,1.,2.],[3.,1.,1.]],requires_grad=True)
    real = torch.tensor([True,False,True])
    y = torch.zeros(3,dtype=torch.long)
    penalty, counts, directions = probability_gap(z,z.detach(),y,y+1,real,dict(margin=.1,weight=1/3))
    (penalty*real).sum().backward()
    assert counts["active"] == counts["rival_switch"] == 1
    assert z.grad[0,0]<0 and z.grad[0,2]>0 and z.grad[0,1]==0
    assert torch.all(z.grad[1:]==0)
    assert torch.isclose(-z.grad[0,0]+z.grad[0,2],directions["margin_gap_gradient"][0])


def test_no_replacement_padded_tail():
    ids = list("abcdefg")
    batches = shuffled_batches(ids,dict(epochs=2,global_batch=3,seed=317))
    orders = [[s["id"] for b in batches if b["epoch"]==e for s in b["slots"] if s["weight"]] for e in (0,1)]
    assert orders[0] != orders[1] and all(sorted(order)==ids for order in orders)
    assert all(len(b["slots"])==3 for b in batches)

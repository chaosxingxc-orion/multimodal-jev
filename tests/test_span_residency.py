import copy
import pytest
import torch
from multimodal_jev.span_residency import SpanResidency


def fixture():
    torch.manual_seed(317)
    model=torch.nn.Linear(3,2)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,foreach=False,fused=False)
    def step(m,o):
        o.zero_grad(set_to_none=True);m(torch.ones(1,3)).square().sum().backward();o.step();o.zero_grad(set_to_none=True)
    step(model,opt)
    session=SpanResidency(model,opt,offload_model=lambda m:None,load_model=lambda m:None,synchronize=lambda:None)
    return model,opt,session,step


def test_cpu_identity_and_next_update_match():
    model,opt,s,step=fixture();other=copy.deepcopy(model)
    ref=torch.optim.AdamW(other.parameters(),lr=1e-4,foreach=False,fused=False);ref.load_state_dict(copy.deepcopy(opt.state_dict()))
    s.offload();assert s.phase=='offloaded';s.reload();assert s.phase=='resident'
    step(model,opt);step(other,ref)
    for a,b in zip(model.parameters(),other.parameters(),strict=True):
        assert torch.equal(a,b)
        for k,v in opt.state[a].items():assert torch.equal(v,ref.state[b][k])


def test_gradients_require_completed_boundary():
    m,o,s,_=fixture();m(torch.ones(1,3)).sum().backward()
    with pytest.raises(ValueError,match='gradients'):s.offload()
    assert s.phase=='resident'


@pytest.mark.parametrize('kind',['parameter','moment','dtype','reference'])
def test_corruption_blocks_reload(kind):
    m,o,s,_=fixture();s.offload();p=next(m.parameters())
    with torch.no_grad():
        if kind=='parameter':p.add_(1)
        elif kind=='moment':o.state[p]['exp_avg'].add_(1)
        elif kind=='dtype':o.state[p]['exp_avg']=o.state[p]['exp_avg'].double()
        else:o.param_groups[0]['params'][0]=torch.nn.Parameter(p.clone())
    with pytest.raises(ValueError):s.reload()
    assert s.phase=='failed'
    with pytest.raises(ValueError):s.reload()


def test_no_implicit_double_offload_or_reload():
    _,_,s,_=fixture()
    with pytest.raises(ValueError):s.reload()
    s.offload()
    with pytest.raises(ValueError):s.offload()
    s.reload()

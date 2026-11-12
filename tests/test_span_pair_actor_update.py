import pytest
import torch
from multimodal_jev.bf16_lora import BF16MasterAdamW
from multimodal_jev.span_actor_update import record_rollout
from multimodal_jev.span_pair_verl import reference_pair_loss
import multimodal_jev.span_pair_actor_update as update


@pytest.mark.parametrize('fault',[None,'stale','input','drift','dtype','pairs'])
def test_update_requires_full_parity_and_changes_bf16_parameters(monkeypatch,fault):
    p=torch.nn.Parameter(torch.tensor([[.3,-.2],[.5,.1],[-.4,.8],[.8,.1]],dtype=torch.bfloat16))
    optimizer=BF16MasterAdamW([p],lr=.01)
    old=tuple(p.detach().float().log_softmax(-1).clone() for _ in range(3))
    ids=['a','b','c','d'];rollout=record_rollout('policy',ids,old,32)
    def forward():
        actor=p.float().log_softmax(-1)
        if fault=='drift':actor=(p.float()+torch.tensor([1.,-1.])).log_softmax(-1)
        return (actor,actor,actor)
    monkeypatch.setattr(update,'verl_pair_loss',lambda *a,**kw:(reference_pair_loss(*a,**kw),{}))
    if fault=='dtype':p.data=p.data.float()
    pair=torch.tensor([[0,2],[3,1]]) if fault!='pairs' else torch.tensor([[0,2],[0,1]])
    before=p.detach().clone()
    args=(rollout,forward,optimizer,[p],torch.tensor([1,0,0,1]),pair,.5,
          'wrong' if fault=='stale' else 'policy',ids[::-1] if fault=='input' else ids)
    if fault:
        with pytest.raises(ValueError):update.update_pair_actor(*args)
        assert torch.equal(before,p) and optimizer.receipt()['optimizer_steps']==[]
    else:
        result=update.update_pair_actor(*args)
        assert result['optimizer_updates']==1 and not torch.equal(before,p)
        assert p.dtype==torch.bfloat16 and optimizer.receipt()['optimizer_steps']==[1]

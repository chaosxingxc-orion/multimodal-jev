import pytest
import torch
from multimodal_jev.context_optimizer_audit import audit_adam_shards


def test_empty_unowned_shard_does_not_require_fabricated_adam_state():
    owned=torch.nn.Parameter(torch.ones(3));empty=torch.nn.Parameter(torch.empty(0))
    opt=torch.optim.AdamW([owned,empty],lr=1e-4)
    owned.sum().backward();opt.step()
    before=len(opt.state)
    report=audit_adam_shards([('owned',owned),('empty',empty)],opt,1)
    assert report['empty']==dict(numel=0,state='unowned_empty')
    assert len(opt.state)==before==1


@pytest.mark.parametrize('fault',['unused_owned','wrong_step','moment_shape','nonfinite'])
def test_owned_parameter_or_moment_failure_is_not_ignored(fault):
    p=torch.nn.Parameter(torch.ones(3));opt=torch.optim.AdamW([p],lr=1e-4)
    if fault!='unused_owned':p.sum().backward();opt.step()
    if fault=='wrong_step':opt.state[p]['step'].fill_(2)
    if fault=='moment_shape':opt.state[p]['exp_avg']=torch.zeros(1)
    if fault=='nonfinite':opt.state[p]['exp_avg'][0]=float('nan')
    with pytest.raises(ValueError):audit_adam_shards([('p',p)],opt,1)

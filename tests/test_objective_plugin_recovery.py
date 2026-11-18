"""Real CPU Adam recovery using the plugin identity stored by the trainer."""
import copy
import hashlib

import pytest
import torch

from multimodal_jev.objective_plugin import ObjectivePlugin
from multimodal_jev.context_training_checkpoint import capture, restore


SOURCE = """API_VERSION = 'jev-stage2-objective-v1'
def compute_rewards(ctx): return None
def compute_loss(ctx, rewards):
    per_query = ctx.logits.square().mean(-1)
    loss = (per_query * ctx.real_requests).sum() * ctx.world_size / ctx.global_real_requests
    return loss * ctx.options['coefficient'], dict(rl=0., auxiliary=0., kl=0.)
"""


def plugin(path, *, coefficient=0.3):
    return ObjectivePlugin(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                options={'coefficient':coefficient}))


def update(parameter, optimizer, objective, step):
    from multimodal_jev.context_dual_credit import prepare, ALGORITHMS
    identities=[dict(policy_id=f'p{step}',input_id='i',query_id='q',update=step)]
    record=prepare(parameter[0].detach(),parameter[0].detach().sigmoid(),['a','b','c'],
                   torch.tensor([True,False,False]),**identities[0],algorithm=ALGORITHMS[1])
    optimizer.zero_grad()
    loss,_=objective.hybrid_loss(parameter,[record],identities,torch.full_like(parameter,.5),
                                  torch.tensor([True]),global_real_requests=1)
    loss.backward();optimizer.step()


def setup(tmp_path):
    path=tmp_path/'objective.py';path.write_text(SOURCE);objective=plugin(path)
    parameter=torch.nn.Parameter(torch.tensor([[.7,-.4,1.2]]))
    optimizer=torch.optim.AdamW([parameter],lr=1e-4,foreach=False)
    update(parameter,optimizer,objective,0)
    identity=dict(rank=0,world_size=1,completed_updates=1,
                  contract=dict(stage='task',objective_plugin=copy.deepcopy(objective.identity)))
    state=capture([('head',parameter)],optimizer,**identity)
    return path,objective,parameter,optimizer,identity,state


def test_same_objective_relocated_file_resumes_exact_next_adam_update(tmp_path):
    path,objective,p,opt,identity,state=setup(tmp_path)
    relocated=tmp_path/'relocated.py';relocated.write_bytes(path.read_bytes())
    replacement=plugin(relocated)
    assert replacement.identity==objective.identity
    update(p,opt,objective,1)
    expected=p.detach().clone();expected_optimizer=copy.deepcopy(opt.state_dict())
    restored=torch.nn.Parameter(torch.zeros_like(p))
    resumed=torch.optim.AdamW([restored],lr=1e-4,foreach=False)
    restore([('head',restored)],resumed,state,**dict(identity,
        contract=dict(stage='task',objective_plugin=replacement.identity)))
    update(restored,resumed,replacement,1)
    assert torch.equal(restored,expected)
    for key,fields in expected_optimizer['state'].items():
        for name,value in fields.items():
            assert torch.equal(value,resumed.state_dict()['state'][key][name])


@pytest.mark.parametrize('change',['file','options','missing_identity'])
def test_changed_objective_rejects_resume_before_parameter_or_adam_mutation(tmp_path,change):
    path,objective,p,opt,identity,state=setup(tmp_path)
    if change=='file':path.write_text(SOURCE+'\n# new objective version\n')
    changed=plugin(path,coefficient=0.7 if change=='options' else 0.3)
    contract=dict(stage='task',objective_plugin=changed.identity)
    if change=='missing_identity':contract.pop('objective_plugin')
    restored=torch.nn.Parameter(torch.zeros_like(p));before=restored.detach().clone()
    resumed=torch.optim.AdamW([restored],lr=1e-4,foreach=False)
    with pytest.raises(ValueError,match='objective'):
        restore([('head',restored)],resumed,state,**dict(identity,contract=contract))
    assert torch.equal(restored,before) and not resumed.state

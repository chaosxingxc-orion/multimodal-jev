"""Exercise generated trainer contract and real Adam checkpoint validation.

No model or NPU equivalence claim: this checks the checkpoint/auditor seam.
"""
import ast
import copy
import json
from pathlib import Path

import pytest
import torch

from multimodal_jev.context_training_checkpoint import capture, restore
from tools.reports.audit_context_macro_training import expected_contract
from tools.reports.context_bucket_trainer_candidate import generate
from tools.reports.context_macro_trainer_candidate import topology


def emitted_contract(c):
    tree=ast.parse(generate(Path.cwd()))
    statements=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='contract' for t in n.targets):
            statements.append(n)
        if (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                and isinstance(n.value.func,ast.Attribute)
                and isinstance(n.value.func.value,ast.Name)
                and n.value.func.value.id=='contract' and n.value.func.attr=='update'):
            statements.append(n)
    assert len(statements)==2
    statements.sort(key=lambda n:n.lineno)
    layout=topology(c,2,['davinci4','davinci5'])
    scope=dict(c=c,world=2,layout=layout,runtime_identity=c['dual_runtime_identity'],
        options=dict(algorithm=c['algorithm'],arm=c['dual_arm'],ranking_coefficient=.1,
            probability_coefficient=.3,kl_coefficient=.01,reference_policy_id=c['source_policy_id']))
    exec(compile(ast.Module(body=statements,type_ignores=[]),'<actual-generated-contract>','exec'),scope)
    return scope['contract']


def config():
    return dict(base_id='base',source_policy_id='policy',source_contract={'token':'fixed'},
        runtime_profile='bf16',release_sha256='release',schedule_sha256='schedule',
        schedule_release_sha256='schedule-release',global_batch=128,microbatch=8,stage='task',
        algorithm='dual_credit_grpo_floor_clipped_v1',behavior_bounds={},dual_arm='joint',
        dual_runtime_identity={'profile':'fixed'},physical_devices=[4,5],
        macro_client=dict(max_queries=128,rollout_queries=32,runtime_identity={'profile':'fixed'}),
        length_buckets=dict(token_bucket_width=128,frame_bucket_width=200,token_cost=1,frame_cost=1))


def test_generated_contract_equals_independent_auditor_after_json_roundtrip():
    c=json.loads(json.dumps(config()))
    assert emitted_contract(c)==dict(expected_contract(c),length_buckets=c['length_buckets'])


@pytest.mark.parametrize('change',[None,'frame_bucket_width','token_cost'])
def test_real_optimizer_restore_requires_same_bucket_strategy(change):
    c=config();contract=emitted_contract(c)
    parameter=torch.nn.Parameter(torch.tensor([.1,.2],dtype=torch.float32))
    def optimizer():
        return torch.optim.AdamW([parameter],lr=1e-4,betas=(.9,.999),eps=1e-8,
            weight_decay=0.,foreach=False,fused=False)
    opt=optimizer();parameter.square().sum().backward();opt.step();opt.zero_grad()
    identity=dict(rank=0,world_size=2,completed_updates=1,contract=contract)
    state=capture([('head',parameter)],opt,**identity)
    with torch.no_grad():parameter.add_(1)
    before=parameter.detach().clone();fresh=optimizer()
    requested=copy.deepcopy(identity)
    if change:
        requested['contract']['length_buckets'][change]+=1
        with pytest.raises(ValueError,match='checkpoint topology'):
            restore([('head',parameter)],fresh,state,**requested)
        assert torch.equal(parameter,before) and not fresh.state
    else:
        result=restore([('head',parameter)],fresh,state,**requested)
        assert result['exact_masters'] and result['exact_adam']
        assert torch.equal(parameter,state['parameters']['head'])

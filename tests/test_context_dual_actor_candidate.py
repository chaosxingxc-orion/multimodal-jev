import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import torch.distributed as dist

from multimodal_jev.context_dual_credit import ALGORITHMS,prepare,hybrid_loss
from multimodal_jev.span_request_identity import tensor_identity
from tools.reports import context_dual_actor_candidate as m


def options(algorithm=ALGORITHMS[0]):
    return dict(algorithm=algorithm,arm='joint',ranking_coefficient=.1,
                probability_coefficient=.3,kl_coefficient=.01,reference_policy_id='fixed')


def items(z=None,algorithm=ALGORITHMS[0]):
    if z is None:z=torch.tensor([.3,-.4])
    rows=[]
    for i in range(4):
        inputs=dict(input_ids=torch.tensor([[i,99,99]]));positions=torch.tensor([[1,2]])
        binding=dict(policy_id='p0',positions=[1,2],inputs={k:tensor_identity(v) for k,v in inputs.items()})
        input_id=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        identity=dict(policy_id='p0',input_id=input_id,query_id=f'q{i}',update=0)
        y=torch.tensor([i%2==0,i%2!=0])
        credit=prepare(z.detach(),z.detach().sigmoid(),['a','b'],y,**identity,algorithm=algorithm)
        rows.append(dict(request_id=f'q{i}',policy_id='p0',inputs=inputs,positions=positions,
            labels=y[None],observed=torch.ones(1,2,dtype=torch.bool),candidate_ids=['a','b'],
            real_request=i!=3,request_weight=float(i!=3),dual_credit=credit,dual_identity=identity,
            reference_probability=torch.tensor([.5,.5]),reference_policy_id='fixed'))
    return rows


def test_native_actor_namespace_isolated_and_reused():
    actor=m.load(Path.cwd())
    from multimodal_jev import context_native_actor as original
    assert actor.MODE!=original.MODE and actor._active is not original._active
    assert actor.make_batch is m.make_batch and actor.admit is m.admit
    assert m.load(Path.cwd()) is actor


@pytest.mark.parametrize('fault',['input','label','candidate','seed','reward','reference','padding'])
def test_dual_item_rechecks_full_binding(fault):
    p=items()[0];o=options()
    assert m.validate_item(p,o).shape==(4,)
    if fault=='input':p['inputs']['input_ids'][0,0]=8
    if fault=='label':p['labels'][0,0]=False
    if fault=='candidate':p['candidate_ids'].reverse()
    if fault=='seed':p['dual_credit']['binding']['seed']=4
    if fault=='reward':p['dual_credit']['branches']['ranking']['rewards'][0,0]+=1
    if fault=='reference':p['reference_policy_id']='wrong'
    if fault=='padding':p['real_request']=False
    with pytest.raises(ValueError):m.validate_item(p,o)


def test_collective_forward_checks_ranking_even_if_existence_rounds_equal(tmp_path):
    dist.init_process_group('gloo',init_method=(tmp_path/'gloo').as_uri(),rank=0,world_size=1)
    try:
        rows=items(torch.tensor([-8.,-10.]))
        m.check_dual_forward(torch.tensor([[-8.,-10.]]*4),rows)
        changed=torch.tensor([[-7.999998,-10.000002]]*4)
        from multimodal_jev.context_behavior_parity import compare
        from tools.reports.context_probability_policy_v2 import binary_logprobs
        old=torch.cat(binary_logprobs(torch.tensor([[-8.,-10.]]*4)),-1)
        new=torch.cat(binary_logprobs(changed),-1)
        assert compare(new,old,dict(probability_max_abs=2e-7,binary_logprob_max_abs=2e-6))['passed']
        # Existence passes both bounds but paired log odds amplify opposing
        # changes. The rank check must independently reject this forward.
        with pytest.raises(ValueError,match='collective dual behavior'):
            m.check_dual_forward(changed,rows)
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('algorithm',ALGORITHMS)
@pytest.mark.parametrize('micro',[1,4])
def test_native_verl_update_matches_whole_batch_gradient(tmp_path,monkeypatch,algorithm,micro):
    pytest.importorskip('verl');pytest.importorskip('omegaconf')
    from probe_span_native_actor_dp_cpu import config
    from multimodal_jev.verl_cpu_logprobs import native_actor_cpu_device
    from torch.nn.parallel import DistributedDataParallel as DDP
    dist.init_process_group('gloo',init_method=(tmp_path/'gloo').as_uri(),rank=0,world_size=1)
    try:
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.z=torch.nn.Parameter(torch.tensor([.3,-.4]))
            def forward(self,n):return self.z[None].expand(n,-1)
        model=Model();wrapped=DDP(model)
        opt=torch.optim.AdamW(wrapped.parameters(),lr=1e-4,weight_decay=0.,foreach=False,fused=False)
        rows=items(model.z,algorithm);opts=options(algorithm)
        actor_module=m.load(Path.cwd())
        calls=[]
        def forward(module,inputs,**kw):
            assert all(set(p)=={'request_id','policy_id','inputs','positions'} for p in inputs)
            calls.append(len(inputs));return module(len(inputs))
        monkeypatch.setattr(actor_module,'forward_context_audio_batch',forward)
        cfg=config(micro,'p0');cfg.ppo_mini_batch_size=4;cfg.policy_loss.loss_mode=m.MODE
        cfg.jev_context_objective=opts;cfg.jev_context_parity=dict(probability_max_abs=2e-7,binary_logprob_max_abs=2e-6)
        batch=m.make_batch(rows,**opts)
        expected=model.z.detach().clone().requires_grad_()
        value,_=hybrid_loss(expected[None].expand(4,-1),[p['dual_credit'] for p in rows],
            [p['dual_identity'] for p in rows],torch.stack([p['reference_probability'] for p in rows]),
            torch.tensor([p['real_request'] for p in rows]),global_real_requests=3)
        grad,=torch.autograd.grad(value,expected)
        actor=actor_module.build_actor(cfg,wrapped,opt,collation={})
        captured=[];original=actor._optimizer_step
        def capture():captured.append(model.z.grad.clone());return original()
        actor._optimizer_step=capture
        with native_actor_cpu_device(actor):actor.update_policy(batch)
        assert calls==[micro]*(4//micro) and len(captured)==1
        torch.testing.assert_close(captured[0],grad,atol=2e-7,rtol=2e-6)
        assert opt.state[model.z]['step'].item()==1
        assert actor.last_admission['global_real_requests']==3
        # One native step only. Corrupt denominator must be rejected collectively.
        bad=copy.deepcopy(batch);bad.non_tensor_batch['multi_modal_inputs'][0]['global_real_requests']=4
        with pytest.raises(ValueError):m.admit(bad,cfg)
    finally:dist.destroy_process_group()

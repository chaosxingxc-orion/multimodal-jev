"""Real four-rank CPU/DDP native verl updates and fresh-process replay.

Synthetic logits isolate distribution/accumulation/optimizer semantics. This
does not substitute for real audio, BF16 FSDP, NPU or engine activation tests.
"""
import copy
from datetime import timedelta
import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from multimodal_jev.context_dual_credit import ALGORITHMS,prepare,hybrid_loss
from multimodal_jev.context_training_checkpoint import capture,restore
from multimodal_jev.span_request_identity import tensor_identity
from tools.reports import context_dual_actor_candidate as adapter


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.z=torch.nn.Parameter(torch.tensor([.3,-.4]))
    def forward(self,n):return self.z[None].expand(n,-1)


def optimizer(model):
    return torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.,foreach=False,fused=False)


def rows_for(z,step,algorithm,total_rows=8,real_rows=5):
    result=[]
    for i in range(total_rows):
        inputs=dict(input_ids=torch.tensor([[i,99,99]]));positions=torch.tensor([[1,2]])
        identity=dict(policy_id=f'p{step}',positions=[1,2],inputs={k:tensor_identity(v) for k,v in inputs.items()})
        rid=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        binding=dict(policy_id=f'p{step}',input_id=rid,query_id=f's{step}/q{i}',update=step)
        labels=torch.tensor([i%3==0,i%3!=0]);real=i<real_rows # remaining slots carry zero objective weight
        credit=prepare(z.detach(),z.detach().sigmoid(),['a','b'],labels,**binding,algorithm=algorithm)
        result.append(dict(request_id=binding['query_id'],policy_id=binding['policy_id'],
            inputs=inputs,positions=positions,labels=labels[None],observed=torch.ones(1,2,dtype=torch.bool),
            candidate_ids=['a','b'],real_request=real,request_weight=float(real),dual_identity=binding,
            dual_credit=credit,reference_probability=torch.tensor([.5,.5]),reference_policy_id='fixed'))
    return result


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal_tree(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b,strict=True):equal_tree(x,y)
    else:assert a==b


def worker(rank,root,replay,world=4,total_rows=8,real_rows=5,micros=(1,2),algorithms=ALGORITHMS):
    from probe_span_native_actor_dp_cpu import config
    from multimodal_jev.verl_cpu_logprobs import native_actor_cpu_device
    root=Path(root);torch.set_num_threads(1);local_rows=total_rows//world
    dist.init_process_group('gloo',init_method=(root/('resume-gloo' if replay else 'continuous-gloo')).as_uri(),
                            rank=rank,world_size=world,timeout=timedelta(seconds=240))
    try:
        module=adapter.load(Path.cwd());cases=[]
        for algorithm in algorithms:
            for micro in micros:
                key=f'{algorithm}-micro{micro}-rank{rank}'
                model=Model();wrapped=DDP(model);opt=optimizer(wrapped)
                named=list(wrapped.named_parameters())
                contract=dict(algorithm=algorithm,micro=micro,global_rows=total_rows,real_rows=real_rows,world=world,scope='synthetic_cpu')
                checkpoint=root/(key+'-step1.pt')
                if replay:
                    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
                    restored=restore(named,opt,saved,rank=rank,world_size=world,completed_updates=1,contract=contract)
                    assert restored['exact_adam'] and restored['exact_masters']
                for step in ([1] if replay else [0,1]):
                    rows=rows_for(model.z,step,algorithm,total_rows,real_rows)
                    options=dict(algorithm=algorithm,arm='joint',ranking_coefficient=.1,
                        probability_coefficient=.3,kl_coefficient=.01,reference_policy_id='fixed')
                    full=adapter.make_batch(rows,**options)
                    incoming=[None];dist.scatter_object_list(incoming,list(full.split(local_rows)) if rank==0 else None,src=0)
                    batch=incoming[0]
                    cfg=config(micro,f'p{step}');cfg.ppo_mini_batch_size=local_rows;cfg.policy_loss.loss_mode=adapter.MODE
                    cfg.jev_context_objective=options;cfg.jev_context_parity=dict(probability_max_abs=2e-7,binary_logprob_max_abs=2e-6)
                    calls=[]
                    def forward(actor,inputs,**kw):
                        assert all(set(p)=={'request_id','policy_id','inputs','positions'} for p in inputs)
                        calls.append(len(inputs));return actor(len(inputs))
                    module.forward_context_audio_batch=forward
                    actor=module.build_actor(cfg,wrapped,opt,collation={})
                    # Exercise the actual actor forward: one rank emits a
                    # saturated probability, peers must fail collectively
                    # before any log-prob exception can strand them.
                    before_bad=copy.deepcopy(model.state_dict());opt_bad=copy.deepcopy(opt.state_dict())
                    def saturated_forward(actor,inputs,**kw):
                        z=actor(len(inputs))
                        return z+100 if rank==world-1 else z
                    module.forward_context_audio_batch=saturated_forward
                    try:
                        with native_actor_cpu_device(actor):actor.update_policy(batch)
                    except ValueError as exc:assert 'collective dual behavior' in str(exc)
                    else:raise AssertionError('rank-local sigmoid saturation accepted')
                    equal_tree(before_bad,model.state_dict());equal_tree(opt_bad,opt.state_dict())
                    module.forward_context_audio_batch=forward
                    # Collective rank-local corruption must reject on every peer.
                    corrupt=copy.deepcopy(batch)
                    if rank==world-1:corrupt.non_tensor_batch['multi_modal_inputs'][0]['global_real_requests']=real_rows+1
                    try:adapter.admit(corrupt,cfg)
                    except ValueError:pass
                    else:raise AssertionError('rank-local denominator corruption accepted')
                    before=copy.deepcopy(model.state_dict());before_opt=copy.deepcopy(opt.state_dict())
                    local_z=model.z.detach()[None].expand(local_rows,-1).clone()
                    if rank==world-1:local_z[0,0]+=.1
                    try:adapter.check_dual_forward(local_z,list(batch.non_tensor_batch['multi_modal_inputs']))
                    except ValueError:pass
                    else:raise AssertionError('one-rank behavior mismatch accepted')
                    equal_tree(before,model.state_dict());equal_tree(before_opt,opt.state_dict())
                    # Independent whole-population aggregation, before native update.
                    reference=model.z.detach().clone().requires_grad_()
                    loss,_=hybrid_loss(reference[None].expand(total_rows,-1),[p['dual_credit'] for p in rows],
                        [p['dual_identity'] for p in rows],torch.stack([p['reference_probability'] for p in rows]),
                        torch.tensor([p['real_request'] for p in rows]),global_real_requests=real_rows)
                    expected,=torch.autograd.grad(loss,reference)
                    grads=[];original=actor._optimizer_step
                    def capture_gradient():grads.append(model.z.grad.clone());return original()
                    actor._optimizer_step=capture_gradient
                    with native_actor_cpu_device(actor):actor.update_policy(batch)
                    assert len(grads)==1 and calls==[micro]*(local_rows//micro)
                    torch.testing.assert_close(grads[0],expected,atol=2e-7,rtol=2e-6)
                    assert opt.state[model.z]['step'].item()==step+1
                    state=capture(named,opt,rank=rank,world_size=world,completed_updates=step+1,contract=contract)
                    samples=[p['dual_credit'] for p in rows]
                    if step==0:torch.save(state,checkpoint)
                    elif not replay:torch.save(dict(state=state,samples=samples,gradient=grads[0]),root/(key+'-expected.pt'))
                    else:
                        target=torch.load(root/(key+'-expected.pt'),weights_only=True)
                        equal_tree(state,target['state']);equal_tree(samples,target['samples']);equal_tree(grads[0],target['gradient'])
                    cases.append(dict(algorithm=algorithm,micro=micro,step=step+1,
                        gradient_max_abs=float((grads[0]-expected).abs().max()),replay_exact=replay))
        (root/f'{"replay" if replay else "continuous"}-rank{rank}.json').write_text(json.dumps(dict(passed=True,cases=cases)))
    finally:dist.destroy_process_group()


def test_four_rank_native_updates_and_fresh_process_resume(tmp_path):
    pytest.importorskip('verl');pytest.importorskip('omegaconf')
    assert not list(Path('/dev').glob('davinci[0-9]*'))
    mp.spawn(worker,args=(str(tmp_path),False),nprocs=4,join=True)
    # New interpreters, process group, module, model and optimizer for replay.
    mp.spawn(worker,args=(str(tmp_path),True),nprocs=4,join=True)
    for rank in range(4):
        a=json.loads((tmp_path/f'continuous-rank{rank}.json').read_text())
        b=json.loads((tmp_path/f'replay-rank{rank}.json').read_text())
        assert a['passed'] and len(a['cases'])==8 and b['passed'] and len(b['cases'])==4
        assert all(c['replay_exact'] for c in b['cases'])

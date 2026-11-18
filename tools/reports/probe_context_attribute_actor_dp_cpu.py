"""Four-rank native verl updates for the prospective native attribute arms.

Uses a synthetic audio backbone and actual shallow Transformer decision head.
The independent reference reconstructs family rewards and the joint loss;
it does not call the candidate objective/reward or verl advantage estimator.
"""
import argparse
import copy
from datetime import timedelta
import json
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from multimodal_jev.baseline import atomic_json
from multimodal_jev.context_audio_collate import forward_context_audio_batch
from multimodal_jev.span_policy import sample_spans, joint_logprobs
from multimodal_jev.verl_cpu_logprobs import native_actor_cpu_device
from tools.reports.context_attribute_actor_candidate import load, REFERENCE_ID
from tools.reports.context_probability_policy_v2 import binary_logprobs
from probe_context_native_actor_dp_cpu import model, optimizer, COLLATION
from probe_span_native_actor_dp_cpu import config, snapshot


def items_for(ref, fixed, step, menu, contract):
    generator=torch.Generator().manual_seed(500+step)
    by_id={n['id']:n for n in menu['nodes']}
    nodes=[by_id[k] for k in contract['decision_ids']]
    result=[]
    for i in range(8):
        frames=100 if i%2==0 else 190
        count=13 if frames==100 else 25
        width=100 if frames==100 else 200
        ids=torch.tensor([[1]+[98]*count+[t for _ in nodes for t in (2,99)]])
        positions=torch.arange(count+2,ids.shape[1],2)[None]
        inputs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),
            input_features=torch.randn(1,4,width,generator=generator),
            input_features_mask=(torch.arange(width)<frames).long()[None])
        event=menu['events'][(i+step*8)%len(menu['events'])]['id']
        labels=torch.tensor([[event in n['members'] for n in nodes]])
        row=dict(request_id=f'{step}/{i}',policy_id=f'policy{step}',inputs=inputs,
            positions=positions,labels=labels,observed=torch.ones_like(labels),
            request_weight=0. if i==7 else 1.,real_request=i!=7,
            attribute_contract_id=contract['contract_id'],decision_ids=contract['decision_ids'])
        transport={k:row[k] for k in ('request_id','policy_id','inputs','positions')}
        with torch.no_grad():
            z=forward_context_audio_batch(ref,[transport],device='cpu',dtype=torch.float32,
                policy_id=row['policy_id'],**COLLATION)
            rz=forward_context_audio_batch(fixed,[transport],device='cpu',dtype=torch.float32,
                policy_id=row['policy_id'],**COLLATION)
        row['behavior']=tuple(binary_logprobs(z))
        row['actions']=sample_spans(row['behavior'],16,generator)[0]
        row['reference_policy']=tuple(binary_logprobs(rz))
        row['reference_policy_id']=REFERENCE_ID
        result.append(row)
    return result


def independent_loss(ref, items, menu, contract, algorithm):
    nodes={n['id']:n for n in menu['nodes']}
    families={f:[i for i,k in enumerate(contract['decision_ids']) if nodes[k]['family']==f]
              for f in ('scenario','action','final')}
    final=families['final']
    inputs=[{k:p[k] for k in ('request_id','policy_id','inputs','positions')} for p in items]
    z=forward_context_audio_batch(ref,inputs,device='cpu',dtype=torch.float32,
        policy_id=items[0]['policy_id'],**COLLATION)
    total=z.sum()*0
    for i,p in enumerate(items):
        y=p['labels'][0,final];v=z[i,final]
        ranking=torch.nn.functional.softplus(v[~y][None]-v[y][:,None]).mean()
        nll=torch.nn.functional.binary_cross_entropy_with_logits(v,y.float())
        scores={}
        for f,indices in families.items():
            if not indices:continue
            a=p['actions'][:,:,indices].float();truth=p['labels'][:,None,indices].float()
            scores[f]=2*(a*truth).sum(-1)/(a.sum(-1)+truth.sum(-1))
        rewards=scores['final']
        if contract['arm']=='q_credit':rewards=rewards+.125*(scores['scenario']+scores['action'])
        centered=rewards-rewards.mean(-1,keepdim=True)
        adv=centered/(rewards.std(-1,keepdim=True)+1e-6) if algorithm=='grpo_v1' else centered*16/15
        policy=binary_logprobs(z[i:i+1])
        old=joint_logprobs(p['behavior'],p['actions']);new=joint_logprobs(policy,p['actions'])
        ratio=(new-old).clamp(-20,20).exp()
        clipped=torch.maximum(-adv*ratio,-adv*ratio.clamp(.8,1.2))
        pg=torch.where(adv<0,torch.minimum(-adv*3,clipped),clipped).mean()
        kl=sum((a.exp()*(a-b)).sum() for a,b in zip(policy,p['reference_policy'],strict=True))
        total=total+p['request_weight']*(pg+.01*kl+.3*ranking+.3*nll)/7
    return total


def worker(rank, output, source_root):
    output=Path(output);source_root=Path(source_root)
    menu=json.loads((source_root/'.local/decision-objective-research/slurp-native-attribute-prob-candidate-v1/catalog.json').read_text())
    torch.set_num_threads(1);torch.manual_seed(317)
    dist.init_process_group('gloo',init_method=(output/'gloo-init').resolve().as_uri(),
        rank=rank,world_size=4,timeout=timedelta(seconds=240))
    try:
        initial=copy.deepcopy(model().state_dict());cases=[]
        for arm in ('direct','q_input','q_credit'):
            native,contract=load(menu,arm,source_root=source_root)
            for algorithm in ('grpo_v1','rloo_v1'):
                for micro in (1,2):
                    bare=model();bare.load_state_dict(initial);wrapped=DDP(bare);opt=optimizer(wrapped)
                    ref=model();ref.load_state_dict(initial);refopt=optimizer(ref)
                    fixed=model();fixed.load_state_dict(initial)
                    for step in range(3):
                        items=items_for(ref,fixed,step,menu,contract)
                        options=dict(stage='task',ranking_coefficient=.3,probability_coefficient=.3,
                            algorithm=algorithm,kl_coefficient=.01,reference_policy_id=REFERENCE_ID,
                            attribute_contract_id=contract['contract_id'])
                        cfg=config(micro,f'policy{step}');cfg.ppo_mini_batch_size=2
                        cfg.policy_loss.loss_mode=native.MODE;cfg.jev_context_objective=options
                        cfg.jev_context_parity=dict(probability_max_abs=2e-6,binary_logprob_max_abs=2e-6)
                        parts=list(native.make_batch(items,**options).split(2)) if rank==0 else None
                        incoming=[None];dist.scatter_object_list(incoming,parts,src=0);batch=incoming[0]
                        refopt.zero_grad();independent_loss(ref,items,menu,contract,algorithm).backward()
                        grads=[p.grad.clone() for p in ref.parameters()]
                        torch.nn.utils.clip_grad_norm_(ref.parameters(),1.);refopt.step()
                        actor=native.build_actor(cfg,wrapped,opt,collation=COLLATION)
                        captured={};original=actor._optimizer_step
                        def capture():
                            captured['grads']=[p.grad.clone() for p in wrapped.parameters()]
                            return original()
                        actor._optimizer_step=capture
                        with native_actor_cpu_device(actor):actor.update_policy(batch)
                        expected=snapshot(ref,refopt,grads);actual=snapshot(wrapped,opt,captured['grads'])
                        errors={k:max(float((a-b).abs().max()) for a,b in zip(v,actual[k],strict=True))
                                for k,v in expected.items()}
                        atomic_json(output/f'comparison-rank-{rank}.json',dict(arm=arm,
                            algorithm=algorithm,micro=micro,step=step,errors=errors))
                        for k,v in expected.items():
                            for a,b in zip(v,actual[k],strict=True):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-4)
                        assert all(opt.state[p]['step'].item()==step+1 for p in wrapped.parameters())
                        if step==0:
                            for fault in ('denominator','mask','padding','duplicate','contract','node_order','reward','advantage'):
                                bad=copy.deepcopy(batch)
                                if fault=='duplicate':
                                    shared=[bad if rank==0 else None];dist.broadcast_object_list(shared,src=0);bad=shared[0]
                                elif rank==3:
                                    row=bad.non_tensor_batch['multi_modal_inputs'][0]
                                    if fault=='denominator':row['global_real_requests']+=1
                                    if fault=='mask':bad.batch['response_mask'][0,0]=0
                                    if fault=='padding':bad.non_tensor_batch['multi_modal_inputs'][-1]['request_weight']=1.
                                    if fault=='contract':row['attribute_contract_id']='wrong'
                                    if fault=='node_order':row['decision_ids']=row['decision_ids'][::-1]
                                    if fault=='reward':row['rewards'][0,0]+=.1
                                    if fault=='advantage':row['advantages'][0,0]+=.1
                                try:native.admit(bad,cfg)
                                except ValueError:pass
                                else:raise AssertionError('corrupt rank batch accepted '+fault)
                        cases.append(dict(arm=arm,algorithm=algorithm,micro=micro,step=step,
                            errors=errors,admission=actor.last_admission,behavior_parity=actor.last_behavior_parity,
                            restored_then_updated=step==2,contract_id=contract['contract_id']))
                        if step==1:
                            path=output/f'restore-{arm}-{algorithm}-{micro}-{rank}.pt'
                            torch.save(dict(model=bare.state_dict(),optimizer=opt.state_dict()),path)
                            with torch.no_grad():
                                for p in bare.parameters():p.add_(1.)
                            saved=torch.load(path,weights_only=True);bare.load_state_dict(saved['model'])
                            opt=optimizer(wrapped);opt.load_state_dict(saved['optimizer'])
        atomic_json(output/f'rank-{rank}.json',dict(passed=True,cases=cases))
    finally:dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True);args=parser.parse_args()
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU-only diagnostic')
    args.output.mkdir(exist_ok=False)
    mp.spawn(worker,args=(str(args.output),str(args.source_root)),nprocs=4,join=True)
    atomic_json(args.output/'result.json',dict(passed=True,ranks=4,cases_per_rank=36,
        scope='Actual native verl/DDP updates, independent gradients/Adam, same-process save/load then update; synthetic audio backbone, actual shared head; no Qwen, new-process resume, NPU or efficacy proof.'))

#!/usr/bin/env python3
"""Pinned native verl actor owns four-rank hybrid update and accumulation."""
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
from multimodal_jev.span_hybrid_actor import MODE, make_batch, build_actor, admit
from multimodal_jev.span_hybrid_objective import sft_coefficient
from multimodal_jev.verl_cpu_logprobs import native_actor_cpu_device
from probe_span_native_actor_dp_cpu import Model, config, optimizer, snapshot
from probe_span_finite_actor_dp_cpu import make_items, effective
from multimodal_jev.span_finite_credit import reference_loss


def reference(model, items, stage, coefficient, real_mean=False, sampled_algorithm=None, kl_coefficient=0.):
    total = sum(i['real_request'] for i in items) if real_mean else sum(effective(i) for i in items)
    for item in items:
        weight = effective(item)
        if not weight:
            continue
        p = model(item['inputs'], item['positions'], item['candidates'])[1]
        y, mask, w = item['targets'], item['supervised'], item['span_weights']
        ce = sum(-v[0, int(y[0, j])]*w[0, j] for j, v in enumerate(p) if mask[0, j])
        if sampled_algorithm:
            actions=item['actions'][:,:,0]
            reward=(actions==y[:,0,None]).float()
            centered=reward-reward.mean(-1,keepdim=True)
            advantage=(centered/(reward.std(-1,keepdim=True)+1e-6) if sampled_algorithm=='grpo_v1'
                       else centered*reward.shape[1]/(reward.shape[1]-1))
            old=item['behavior'][0].gather(1,actions);new=p[0].gather(1,actions)
            ratio=(new-old).clamp(-20,20).exp()
            clipped=torch.maximum(-advantage*ratio,-advantage*ratio.clamp(.8,1.2))
            rl=torch.where(advantage<0,torch.minimum(-advantage*3,clipped),clipped).mean()
            kl=sum((v.exp()*(v-q)).sum() for v,q in zip(p,item['reference_policy'],strict=True)) if kl_coefficient else rl*0
            value=rl+coefficient*ce+kl_coefficient*kl
        else:
            value = ce if stage == 'instruction' else reference_loss(item['behavior'], p, y, mask, w) + coefficient*ce
        (value*weight/total).backward()


def worker(rank, output, real_mean=False, sampled_algorithm=None, kl_coefficient=0., rollout_bridge=False):
    output = Path(output); torch.set_num_threads(1); torch.manual_seed(317)
    if list(Path('/dev').glob('davinci[0-9]*')):
        raise ValueError('CPU-only')
    dist.init_process_group('gloo', init_method=(output/'init').resolve().as_uri(), rank=rank,
                            world_size=4, timeout=timedelta(seconds=180))
    try:
        initial = copy.deepcopy(Model().state_dict()); cases = []
        for micro in (1, 2):
            module = Model(); module.load_state_dict(initial); model = DDP(module); opt = optimizer(model)
            ref = Model(); ref.load_state_dict(initial); refopt = optimizer(ref)
            cfg = config(micro, 'step0'); cfg.ppo_mini_batch_size = 2; cfg.policy_loss.loss_mode = MODE
            fixed_reference=Model();fixed_reference.load_state_dict(initial)
            cfg.jev_advantage_algorithm=sampled_algorithm or 'exact_marginal_legacy'
            cfg.jev_kl_coefficient=kl_coefficient
            cfg.jev_reference_policy_id='stage1-fixed' if kl_coefficient else None
            actor = build_actor(cfg, model, opt)
            for step in range(3):
                stage = 'instruction' if step == 0 or real_mean else 'task'
                if sampled_algorithm:stage='task'
                coefficient = sft_coefficient(step,3) if sampled_algorithm else 0. if stage == 'instruction' else sft_coefficient(step-1, 2)
                items = make_items(ref, step)
                if sampled_algorithm:
                    for item in items:
                        item['candidates']=[list(range(60))] if rollout_bridge else item['candidates'][:1]
                        item['positions']=item['positions'][:,:1]
                        item['targets']=item['targets'][:,:1].clamp_min(0)
                        item['supervised']=torch.ones_like(item['targets'],dtype=torch.bool)
                        item['span_weights']=torch.ones_like(item['targets'],dtype=torch.float32)
                        with torch.no_grad():item['behavior']=ref(item['inputs'],item['positions'],item['candidates'])[1]
                        if not rollout_bridge:
                            size=len(item['candidates'][0]);target=int(item['targets'][0,0])
                            values=[target if i%3==0 else (target+1)%size for i in range(16)]
                            item['actions']=torch.tensor(values)[None,:,None]
                        if kl_coefficient:
                            with torch.no_grad():item['reference_policy']=fixed_reference(item['inputs'],item['positions'],item['candidates'])[1]
                            item['reference_policy_id']='stage1-fixed'
                    if not rollout_bridge:
                        items[1]['actions'].fill_((int(items[1]['targets'][0,0])+1)%len(items[1]['candidates'][0]))
                items[6]['request_weight'] = 0.
                if real_mean:
                    for index, item in enumerate(items):
                        item['real_request'] = index != 6
                        if index != 6:
                            item['request_weight'] = .5 if index % 2 else 1.5
                cfg.jev_policy_id = f'policy{step}'
                sampling_receipts=[]
                if rollout_bridge:
                    import hashlib
                    from multimodal_jev.simple_sampled_rollout import actor_item
                    from multimodal_jev.span_request_identity import tensor_identity
                    bridged=[]
                    for slot,item in enumerate(items):
                        identity=dict(request={'audio':f'synthetic:{slot}'},
                            inputs={k:tensor_identity(v) for k,v in sorted(item['inputs'].items())},
                            positions=item['positions'].tolist(),candidate_ids=item['candidates'],
                            sampling=dict(temperature=1.0,policy='independent_finite_categorical'))
                        records=[]
                        for policy,probs in [(item['policy_id'],item['behavior']),('stage1-fixed',item['reference_policy'])]:
                            wire=dict(identity,policy_id=policy)
                            rid=hashlib.sha256(json.dumps(wire,sort_keys=True,separators=(',', ':')).encode()).hexdigest()
                            records.append(dict(request_id=rid,request_identity=wire,candidate_logprobs=[p[0].tolist() for p in probs]))
                        packed,receipt=actor_item(item,*records,expected_identity=records[0]['request_identity'],
                            reference_policy_id='stage1-fixed',seed=317,update=step,slot=slot)
                        bridged.append(packed);sampling_receipts.append(receipt)
                    items=bridged
                refopt.zero_grad(); reference(ref, items, stage, coefficient, real_mean, sampled_algorithm, kl_coefficient)
                grads = [p.grad.clone() for p in ref.parameters()]
                torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.); refopt.step()
                if stage == 'instruction':
                    items = [{k:v for k,v in item.items() if k != 'behavior'} for item in items]
                chunks = list(make_batch(items, stage=stage, coefficient=coefficient,
                    reduction='real_request_mean' if real_mean else 'legacy_weight_mean',
                    algorithm=sampled_algorithm or 'exact_marginal_legacy',kl_coefficient=kl_coefficient,
                    reference_policy_id='stage1-fixed' if kl_coefficient else None).split(2)) if rank == 0 else None
                incoming = [None]; dist.scatter_object_list(incoming, chunks, src=0); batch = incoming[0]
                captured = {}; original = actor._optimizer_step
                def capture():
                    captured['gradients'] = [p.grad.clone() for p in model.parameters()]
                    return original()
                actor._optimizer_step = capture
                with native_actor_cpu_device(actor):
                    metrics = actor.update_policy(batch)
                actor._optimizer_step = original
                expected = snapshot(ref, refopt, grads); actual = snapshot(model, opt, captured['gradients'])
                errors = {}
                for name, values in expected.items():
                    errors[name] = max(float((a-b).abs().max()) for a,b in zip(values, actual[name], strict=True))
                    for a,b in zip(values, actual[name], strict=True):
                        torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-5)
                rejections = []
                for defect in ('denominator', 'mask', 'old_probability', 'coefficient', 'duplicate') + (('real_count', 'padding') if real_mean else ()) + (('advantage','algorithm','sampled_old') if sampled_algorithm else ()):
                    bad = copy.deepcopy(batch)
                    if defect == 'duplicate':
                        same = [bad if rank == 0 else None]; dist.broadcast_object_list(same, src=0); bad = same[0]
                    elif rank == 3:
                        if defect == 'denominator':
                            bad.non_tensor_batch['multi_modal_inputs'][0]['global_weight'] += 1
                        elif defect == 'coefficient':
                            bad.non_tensor_batch['multi_modal_inputs'][0]['coefficient'] += .1
                        elif defect == 'mask':
                            bad.batch['response_mask'][0, 0] = 0
                        elif defect == 'real_count':
                            bad.non_tensor_batch['multi_modal_inputs'][0]['global_real_requests'] += 1
                        elif defect == 'advantage':
                            bad.non_tensor_batch['multi_modal_inputs'][0]['sampled_advantages'][0,0] += .1
                        elif defect == 'algorithm':
                            bad.non_tensor_batch['multi_modal_inputs'][0]['algorithm']='unknown'
                        elif defect == 'sampled_old':
                            bad.non_tensor_batch['multi_modal_inputs'][0]['old_joint'][0,0] += .1
                        elif defect == 'padding':
                            bad.non_tensor_batch['multi_modal_inputs'][0]['real_request'] = True
                        else:
                            bad.batch['old_log_probs'][0, 0] += .1
                    try:
                        admit(bad, cfg)
                    except ValueError:
                        rejections.append(defect)
                    else:
                        raise AssertionError('bad hybrid transport accepted: '+defect)
                cases.append(dict(micro=micro, step=step, stage=stage, coefficient=coefficient,
                    errors=errors, metrics=metrics, admission=actor.last_admission, rejected=rejections,
                    sampling_receipts=sampling_receipts))
                if step == 1:
                    checkpoint = output/f'checkpoint-{rank}-{micro}.pt'
                    torch.save(dict(model=model.module.state_dict(), optimizer=opt.state_dict(), next_step=2), checkpoint)
                    state = torch.load(checkpoint, weights_only=True)
                    model.module.load_state_dict(state['model']); opt = optimizer(model); opt.load_state_dict(state['optimizer'])
                    actor = build_actor(cfg, model, opt)
                    assert state['next_step'] == step+1
            del actor, model
        atomic_json(output/f'rank-{rank}.json', dict(passed=True, rank=rank, cases=cases))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--real-request-mean',action='store_true')
    p.add_argument('--sampled-algorithm',choices=('grpo_v1','rloo_v1'))
    p.add_argument('--kl-coefficient',type=float,default=0.)
    p.add_argument('--rollout-bridge',action='store_true');a=p.parse_args()
    if a.kl_coefficient and not a.sampled_algorithm:p.error('KL requires sampled algorithm')
    if a.rollout_bridge and (not a.sampled_algorithm or a.kl_coefficient<=0):p.error('bridge requires sampled algorithm and fixed reference KL')
    if a.sampled_algorithm:a.real_request_mean=True
    a.output.mkdir(exist_ok=False,parents=True)
    try:
        mp.spawn(worker,args=(str(a.output),a.real_request_mean,a.sampled_algorithm,a.kl_coefficient,a.rollout_bridge),nprocs=4,join=True)
        ranks=[json.loads((a.output/f'rank-{i}.json').read_text()) for i in range(4)]
        atomic_json(a.output/'result.json',dict(passed=True,ranks=ranks,real_request_mean=a.real_request_mean,algorithm=a.sampled_algorithm,kl_coefficient=a.kl_coefficient,rollout_bridge=a.rollout_bridge,
            scope='Actual pinned native verl actor update loop, four Gloo/DDP CPU ranks, mixed candidate CE/PPO, accumulation and Adam restore. Synthetic FP32 only; no NPU/audio/vLLM efficacy claim.'))
        atomic_json(a.output/'terminal.json',dict(exit_code=0))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise

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


def reference(model, items, stage, coefficient):
    total = sum(effective(i) for i in items)
    for item in items:
        weight = effective(item)
        if not weight:
            continue
        p = model(item['inputs'], item['positions'], item['candidates'])[1]
        y, mask, w = item['targets'], item['supervised'], item['span_weights']
        ce = sum(-v[0, int(y[0, j])]*w[0, j] for j, v in enumerate(p) if mask[0, j])
        value = ce if stage == 'instruction' else reference_loss(item['behavior'], p, y, mask, w) + coefficient*ce
        (value*weight/total).backward()


def worker(rank, output):
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
            actor = build_actor(cfg, model, opt)
            for step in range(3):
                stage = 'instruction' if step == 0 else 'task'
                coefficient = 0. if step == 0 else sft_coefficient(step-1, 2)
                items = make_items(ref, step)
                items[6]['request_weight'] = 0.
                cfg.jev_policy_id = f'policy{step}'
                refopt.zero_grad(); reference(ref, items, stage, coefficient)
                grads = [p.grad.clone() for p in ref.parameters()]
                torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.); refopt.step()
                if stage == 'instruction':
                    items = [{k:v for k,v in item.items() if k != 'behavior'} for item in items]
                chunks = list(make_batch(items, stage=stage, coefficient=coefficient).split(2)) if rank == 0 else None
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
                for defect in ('denominator', 'mask', 'old_probability', 'coefficient', 'duplicate'):
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
                        else:
                            bad.batch['old_log_probs'][0, 0] += .1
                    try:
                        admit(bad, cfg)
                    except ValueError:
                        rejections.append(defect)
                    else:
                        raise AssertionError('bad hybrid transport accepted: '+defect)
                cases.append(dict(micro=micro, step=step, stage=stage, coefficient=coefficient,
                    errors=errors, metrics=metrics, admission=actor.last_admission, rejected=rejections))
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
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False,parents=True)
    try:
        mp.spawn(worker,args=(str(a.output),),nprocs=4,join=True)
        ranks=[json.loads((a.output/f'rank-{i}.json').read_text()) for i in range(4)]
        atomic_json(a.output/'result.json',dict(passed=True,ranks=ranks,
            scope='Actual pinned native verl actor update loop, four Gloo/DDP CPU ranks, mixed candidate CE/PPO, accumulation and Adam restore. Synthetic FP32 only; no NPU/audio/vLLM efficacy claim.'))
        atomic_json(a.output/'terminal.json',dict(exit_code=0))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise

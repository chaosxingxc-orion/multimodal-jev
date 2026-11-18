#!/usr/bin/env python3
"""Exercise continuous driver with actual pinned verl on four Gloo CPU ranks.

Synthetic FP32 full60/G16 control, independent explicit loss/Adam comparison.
No NPU/audio/vLLM or formal training admission claim.
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
from multimodal_jev.simple_rl_loop import run
from multimodal_jev.span_training_session import SpanTrainingSession
from multimodal_jev.span_hybrid_actor import MODE, make_batch, build_actor
from multimodal_jev.span_policy import sample_spans
from multimodal_jev.verl_cpu_logprobs import native_actor_cpu_device
from probe_span_native_actor_dp_cpu import Model, config, optimizer, snapshot
from probe_span_finite_actor_dp_cpu import make_items
from probe_span_hybrid_actor_dp_cpu import reference


class Schedule:
    total_updates = 3
    sha256 = 'synthetic-three-update-native-driver-v1'
    def at(self, step):
        return [step], dict(update=step, next_optimizer_update=step+1, total_updates=3,
            coefficient=1.-.3*step, real_requests=7, batch_index=step, schedule_sha256=self.sha256)


def worker(rank, output):
    torch.set_num_threads(1); torch.manual_seed(317)
    output = Path(output)
    if list(Path('/dev').glob('davinci[0-9]*')): raise ValueError('CPU only')
    dist.init_process_group('gloo', init_method=(output/'gloo-init').resolve().as_uri(),
        rank=rank, world_size=4, timeout=timedelta(seconds=180))
    try:
        def gather(value):
            peers = [None]*4; dist.all_gather_object(peers, value); return peers
        initial = copy.deepcopy(Model().state_dict()); reports = []
        for algorithm in ('grpo_v1', 'rloo_v1'):
            for micro in (1, 2):
                module = Model(); module.load_state_dict(initial); model = DDP(module); opt = optimizer(model)
                ref = Model(); ref.load_state_dict(initial); refopt = optimizer(ref)
                fixed = Model(); fixed.load_state_dict(initial)
                cfg = config(micro, 'policy0'); cfg.ppo_mini_batch_size = 2
                cfg.policy_loss.loss_mode = MODE; cfg.jev_advantage_algorithm = algorithm
                cfg.jev_kl_coefficient = .01; cfg.jev_reference_policy_id = 'fixed'
                actor = build_actor(cfg, model, opt)
                session = SpanTrainingSession(actor, policy_id='policy0', completed_updates=0, gather=gather)
                expected = {}; captured = {}; records = []; checkpoint_states = []

                def install_capture(actor):
                    original = actor._optimizer_step
                    def capture():
                        captured['gradients'] = [p.grad.detach().clone() for p in model.parameters()]
                        return original()
                    actor._optimizer_step = capture
                install_capture(actor)

                def collect(slots, cursor, policy):
                    step = cursor['update']; items = make_items(ref, step)
                    for index, item in enumerate(items):
                        item.update(policy_id=policy, candidates=[list(range(60))],
                            positions=item['positions'][:,:1], targets=item['targets'][:,:1].clamp_min(0),
                            supervised=torch.ones(1,1,dtype=torch.bool), span_weights=torch.ones(1,1),
                            real_request=index != 6, request_weight=0. if index == 6 else (.5 if index%2 else 1.5),
                            reference_policy_id='fixed')
                        with torch.no_grad():
                            item['behavior'] = ref(item['inputs'], item['positions'], item['candidates'])[1]
                            item['reference_policy'] = fixed(item['inputs'], item['positions'], item['candidates'])[1]
                        item['actions'], _ = sample_spans(item['behavior'], 16,
                            torch.Generator().manual_seed(317+step*8+index))
                    # Explicit reference objective, separate from native loss callback.
                    refopt.zero_grad()
                    reference(ref, items, 'task', cursor['coefficient'], True, algorithm, .01)
                    gradients = [p.grad.clone() for p in ref.parameters()]
                    torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.); refopt.step()
                    expected.update(snapshot(ref, refopt, gradients))
                    chunks = list(make_batch(items, stage='task', coefficient=cursor['coefficient'],
                        reduction='real_request_mean', algorithm=algorithm, kl_coefficient=.01,
                        reference_policy_id='fixed').split(2)) if rank == 0 else None
                    incoming = [None]; dist.scatter_object_list(incoming, chunks, src=0)
                    return incoming[0]

                def export(session, batch, metrics):
                    actual = snapshot(model, opt, captured['gradients'])
                    for key, values in expected.items():
                        for a,b in zip(values, actual[key], strict=True):
                            torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-5)
                    return 'policy'+str(session.completed_updates)

                def verify(policy, updates, batch):
                    for parameter in model.parameters():
                        assert opt.state[parameter]['step'].item() == updates
                    with torch.no_grad():
                        for payload in batch.non_tensor_batch['multi_modal_inputs']:
                            actual = model(payload['inputs'],payload['positions'],payload['candidates'])[1]
                            correct = ref(payload['inputs'],payload['positions'],payload['candidates'])[1]
                            for a,b in zip(actual,correct,strict=True):
                                torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-5)
                    return dict(passed=True,policy_id=policy,updates=updates,reference='explicit-loss-and-probabilities')

                def checkpoint(session, cursor):
                    state = dict(model=model.module.state_dict(), optimizer=opt.state_dict(),
                        cursor=session.checkpoint_cursor(schedule_sha256=Schedule.sha256,checkpoint_id='cpu-fixture'))
                    path = output/f'{algorithm}-micro{micro}-rank{rank}-step{session.completed_updates}.pt'
                    torch.save(state,path); checkpoint_states.append(path)
                    return dict(updates=session.completed_updates)

                callbacks = dict(collect=collect,export=export,verify=verify,checkpoint=checkpoint,
                                 checkpoint_interval=2,record=records.append)
                with native_actor_cpu_device(actor): run(session,Schedule(),stop_update=2,**callbacks)
                saved = torch.load(checkpoint_states[-1],weights_only=True)
                model.module.load_state_dict(saved['model']); opt = optimizer(model);opt.load_state_dict(saved['optimizer'])
                actor = build_actor(cfg,model,opt);install_capture(actor)
                session = SpanTrainingSession.restore(actor,saved['cursor'],schedule_sha256=Schedule.sha256,
                    checkpoint_id='cpu-fixture',gather=gather)
                with native_actor_cpu_device(actor): result = run(session,Schedule(),stop_update=3,**callbacks)
                if len(records)!=3 or result['completed_updates']!=3: raise ValueError('incomplete native loop')
                reports.append(dict(algorithm=algorithm,micro=micro,records=records,restored_after_update2=True))
        atomic_json(output/f'rank-{rank}.json',dict(passed=True,rank=rank,reports=reports))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    try:
        mp.spawn(worker,args=(str(a.output),),nprocs=4,join=True)
        atomic_json(a.output/'result.json',dict(passed=True,
            ranks=[json.loads((a.output/f'rank-{r}.json').read_text()) for r in range(4)],
            scope='Continuous native verl GRPO/RLOO+CE/KL driver, real Gloo DP4, micro1/2, explicit numerical reference and Adam reload; synthetic CPU only.'))
        atomic_json(a.output/'terminal.json',dict(exit_code=0))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise

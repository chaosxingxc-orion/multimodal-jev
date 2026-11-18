#!/usr/bin/env python3
"""Native verl DP4/global128 true-batch update equivalence on CPU fixtures."""
import argparse
import copy
from datetime import timedelta
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from multimodal_jev.span_batched_hybrid_actor import build_batched_actor
from multimodal_jev.span_hybrid_actor import MODE, build_actor, make_batch
from multimodal_jev.verl_cpu_logprobs import native_actor_cpu_device
from probe_span_native_actor_dp_cpu import config, optimizer, snapshot


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 8)
        self.head = torch.nn.Linear(8, 60, bias=False)
        self.calls = 0

    def forward(self, inputs, positions, candidates):
        self.calls += 1
        audio = (inputs['input_features'] * inputs['input_features_mask'][:, None]).sum((1, 2))
        hidden = self.embedding(inputs['input_ids']) + audio[:, None, None] * .001
        hidden = (hidden * inputs['attention_mask'][..., None]).cumsum(1)
        selected = hidden.gather(1, positions[..., None].expand(-1, -1, 8))
        logits = self.head(selected)
        return logits, tuple(logits[:, j, c].log_softmax(-1) for j, c in enumerate(candidates))


def items_for(model, algorithm):
    items = []
    for i in range(128):
        frames = [31, 100, 205][i % 3]
        full, tail = divmod(frames, 100)
        ids = torch.tensor([[1] + [9] * (full * 13 + (tail+7)//8) + [2, 3]])
        width = ((frames+99)//100)*100
        inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
            input_features=torch.ones(1, 2, width) * (i % 7 + 1) / 100,
            input_features_mask=(torch.arange(width)[None] < frames).long())
        item = dict(request_id=f'learning{i}', policy_id='fixed', inputs=inputs,
            positions=torch.tensor([[ids.shape[1]-1]]), candidates=[list(range(60))],
            targets=torch.tensor([[i % 60]]), supervised=torch.tensor([[True]]),
            span_weights=torch.ones(1, 1), request_weight=0. if i == 127 else .5+(i % 3)/2,
            real_request=i != 127)
        if algorithm != 'ce':
            with torch.no_grad():
                item['behavior'] = tuple(v.detach() for v in model(inputs, item['positions'], item['candidates'])[1])
            item['reference_policy'] = item['behavior']
            item['reference_policy_id'] = 'reference'
            # Mixed groups and constant groups both remain in the real denominator.
            item['actions'] = torch.tensor([[[i % 60 if j % 3 == 0 and i % 5 else (i+1) % 60]
                                            for j in range(16)]])
        items.append(item)
    return items


def worker(rank, root):
    root = Path(root); torch.set_num_threads(1); torch.manual_seed(317)
    if list(Path('/dev').glob('davinci[0-9]*')):
        raise ValueError('CPU-only fixture')
    dist.init_process_group('gloo', init_method=(root/'init').resolve().as_uri(),
                            rank=rank, world_size=4, timeout=timedelta(seconds=240))
    try:
        initial = Model().state_dict(); results = []
        for algorithm in ('ce', 'grpo_v1', 'rloo_v1'):
            template = Model(); template.load_state_dict(initial)
            items = items_for(template, algorithm)
            baseline = None
            for micro in (0, 1, 2, 4, 8, 16, 32):
                module = Model(); module.load_state_dict(initial)
                net = DDP(module); opt = optimizer(net)
                cfg = config(micro or 1, 'fixed'); cfg.ppo_mini_batch_size = 32
                cfg.policy_loss.loss_mode = MODE
                sampled = algorithm != 'ce'
                cfg.jev_advantage_algorithm = algorithm if sampled else 'exact_marginal_legacy'
                cfg.jev_kl_coefficient = .01 if sampled else 0.
                cfg.jev_reference_policy_id = 'reference' if sampled else None
                actor = (build_batched_actor(cfg, net, opt, collation=dict(pad_token_id=0,
                    audio_token_id=9, audio_chunk_size=100, audio_infer_window=800))
                    if micro else build_actor(cfg, net, opt))
                chunks = list(make_batch(items, stage='task' if sampled else 'instruction',
                    coefficient=.4 if sampled else 0., reduction='real_request_mean',
                    algorithm=cfg.jev_advantage_algorithm, kl_coefficient=cfg.jev_kl_coefficient,
                    reference_policy_id=cfg.jev_reference_policy_id).split(32)) if rank == 0 else None
                incoming = [None]; dist.scatter_object_list(incoming, chunks, src=0)
                capture = {}; original = actor._optimizer_step
                def step():
                    capture['gradients'] = [p.grad.clone() for p in net.parameters()]
                    return original()
                actor._optimizer_step = step
                with native_actor_cpu_device(actor):
                    actor.update_policy(incoming[0])
                actual = snapshot(net, opt, capture['gradients'])
                assert module.calls == 32 // (micro or 1)
                if baseline is None:
                    baseline = copy.deepcopy(actual)
                errors = {}
                for key in baseline:
                    errors[key] = max(float((a-b).abs().max()) for a,b in zip(actual[key], baseline[key], strict=True))
                    for a,b in zip(actual[key], baseline[key], strict=True):
                        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
                results.append(dict(algorithm=algorithm, microbatch=micro or 1,
                    path='batched' if micro else 'serial', forward_calls=module.calls, errors=errors))
        (root/f'rank-{rank}.json').write_text(json.dumps(dict(passed=True, rank=rank, results=results), indent=2)+'\n')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); a = p.parse_args()
    root = Path(a.output); root.mkdir(parents=True, exist_ok=False)
    mp.spawn(worker, args=(str(root),), nprocs=4, join=True)
    print(json.dumps(dict(passed=True, ranks=4, global_batch=128, real_requests=127,
        microbatches=[1, 2, 4, 8, 16, 32], algorithms=['ce','grpo_v1','rloo_v1'],
        scope='Native verl four-rank Gloo/DDP, synthetic audio backbone, FP32; not pretrained BF16/FSDP/NPU admission.')))

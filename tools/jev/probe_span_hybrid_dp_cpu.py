#!/usr/bin/env python3
"""Real four-rank mixed-loss gradient and Adam continuation, CPU synthetic only."""
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
from multimodal_jev.span_hybrid_objective import loss, sft_coefficient
from multimodal_jev.span_finite_credit import reference_loss


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 60, bias=False)

    def forward(self, inputs):
        z = self.head(inputs)
        return (z[:, 0].log_softmax(-1), z[:, 1, :5].log_softmax(-1), z[:, 2, :2].log_softmax(-1))


def data(step):
    gen = torch.Generator().manual_seed(431+step)
    x = torch.randn(8, 3, 3, generator=gen)
    y = torch.tensor([[(i+step) % 60, i % 5, i % 2] for i in range(8)])
    mask = torch.ones(8, 3, dtype=torch.bool); mask[4:6] = False; mask[1, 1] = False
    y[~mask] = -100
    weights = torch.tensor([[1., .1, .1]]).repeat(8, 1)
    request_weights = torch.tensor([1., 2., 1., 3., 1., 1., 0., .5])
    return x, y, mask, weights, request_weights


def reference(model, inputs, old, stage, coefficient):
    x, y, mask, w, rw = inputs
    p = model(x)
    active = rw*(mask*w).sum(-1).gt(0)
    denom = active.sum()
    ce = sum((-v.gather(1, y[:, j].masked_fill(~mask[:, j], 0)[:, None])[:, 0]
              *w[:, j]*mask[:, j]*active).sum() for j, v in enumerate(p))/denom
    if stage == 'instruction':
        return ce
    return reference_loss(old, p, y, mask, w, rw) + coefficient*ce


def worker(rank, output):
    torch.set_num_threads(1); torch.manual_seed(317)
    output = Path(output)
    if list(Path('/dev').glob('davinci[0-9]*')):
        raise ValueError('CPU-only probe')
    dist.init_process_group('gloo', init_method=(output/'init').resolve().as_uri(),
                            rank=rank, world_size=4, timeout=timedelta(seconds=180))
    try:
        initial = Model().state_dict(); reports = []
        for micro in (1, 2):
            model = DDP(Model()); model.module.load_state_dict(initial)
            ref = Model(); ref.load_state_dict(initial)
            opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0., foreach=False)
            refopt = torch.optim.AdamW(ref.parameters(), lr=.001, weight_decay=0., foreach=False)
            for step in range(3):
                inputs = data(step); x, y, mask, w, rw = inputs
                stage = 'instruction' if step == 0 else 'task'
                coefficient = 0. if step == 0 else sft_coefficient(step-1, 2)
                with torch.no_grad(): old = tuple(v.detach() for v in ref(x))
                refopt.zero_grad(); reference(ref, inputs, old, stage, coefficient).backward()
                expected_grad = [p.grad.clone() for p in ref.parameters()]
                torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.); refopt.step()
                opt.zero_grad()
                local = slice(rank*2, rank*2+2)
                denom = (rw[local]*(w[local]*mask[local]).sum(-1).gt(0)).sum()
                dist.all_reduce(denom)
                for start in range(rank*2, rank*2+2, micro):
                    sl = slice(start, start+micro)
                    new = model(x[sl])
                    value, metrics = loss(new, y[sl], mask[sl], w[sl], rw[sl], stage=stage,
                        global_effective_weight=float(denom), world_size=4,
                        behavior=None if stage == 'instruction' else tuple(v[sl] for v in old), coefficient=coefficient)
                    value.backward()
                gradient_error = max(float((p.grad-g).abs().max()) for p, g in zip(model.parameters(), expected_grad))
                for p, g in zip(model.parameters(), expected_grad):
                    torch.testing.assert_close(p.grad, g, atol=2e-7, rtol=2e-5)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
                for p, q in zip(model.parameters(), ref.parameters()):
                    torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-5)
                    for name in ('exp_avg', 'exp_avg_sq'):
                        torch.testing.assert_close(opt.state[p][name], refopt.state[q][name], atol=2e-7, rtol=2e-5)
                reports.append(dict(micro=micro, update=step+1, stage=stage,
                    coefficient=coefficient, gradient_max_error=gradient_error, global_weight=float(denom)))
                if step == 1:
                    # Exercise real file restore before the following update, not just tensor equality.
                    path = output/f'checkpoint-rank{rank}-micro{micro}.pt'
                    torch.save(dict(model=model.module.state_dict(), optimizer=opt.state_dict(), next_update=2), path)
                    state = torch.load(path, weights_only=True)
                    model.module.load_state_dict(state['model'])
                    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0., foreach=False)
                    opt.load_state_dict(state['optimizer'])
                    assert state['next_update'] == step+1
            del model
        atomic_json(output/f'rank-{rank}.json', dict(passed=True, rank=rank, cases=reports))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    a.output.mkdir(exist_ok=False, parents=True)
    try:
        mp.spawn(worker, args=(str(a.output),), nprocs=4, join=True)
        ranks = [json.loads((a.output/f'rank-{i}.json').read_text()) for i in range(4)]
        atomic_json(a.output/'result.json', dict(passed=True, ranks=ranks,
            scope='Real Gloo DDP, native verl loss, synthetic FP32 mixed objective, global normalization and Adam resume. No native actor lifecycle, BF16/audio/NPU or vLLM acceptance.'))
        atomic_json(a.output/'terminal.json', dict(exit_code=0))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json', dict(exit_code=1, error=repr(exc))); raise

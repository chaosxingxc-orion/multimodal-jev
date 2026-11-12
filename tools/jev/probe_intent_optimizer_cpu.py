#!/usr/bin/env python3
"""CPU optimizer-control check; not FSDP, audio execution or efficacy evidence."""
import argparse
import copy
import json
from pathlib import Path

import torch
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.intent_optimizer import optimizer_kwargs, validate_saved_optimizer
from multimodal_jev.span_fsdp_resume import restore_local_shard


def model():
    result = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        result.weight.copy_(torch.tensor([[.01, -.02, .03]]).bfloat16().float())
    return result


def snapshot(net, opt, updates):
    return dict(rank=0, world_size=1, updates=updates,
                names=[n for n, p in net.named_parameters()],
                parameters={n: p.detach().clone() for n, p in net.named_parameters()},
                optimizer=copy.deepcopy(opt.state_dict()))


def equal(a, b):
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)): return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def update(net, opt, gradient):
    net.weight.grad = torch.tensor([gradient])
    opt.step(); opt.zero_grad(set_to_none=True)


def run():
    if list(Path('/dev').glob('davinci[0-9]*')): raise ValueError('CPU-only probe')
    torch.set_num_threads(2)
    gradients = [[1., -1., 1.], [-.01, .01, -.01], [.3, -.4, .2]]
    default = dict(lr=1e-4)
    a, b = model(), model()
    oa = torch.optim.AdamW(a.parameters(), lr=1e-4, weight_decay=0., foreach=False, fused=False)
    ob = torch.optim.AdamW(b.parameters(), **optimizer_kwargs(default))
    for step, g in enumerate(gradients, 1):
        update(a, oa, g); update(b, ob, g)
        if not equal(snapshot(a, oa, step), snapshot(b, ob, step)):
            raise AssertionError('explicit defaults changed the original trajectory')
    reports = []
    for beta1 in (.9, 0.):
        config = dict(lr=1e-4, adam_betas=[beta1, .999])
        net = model(); opt = torch.optim.AdamW(net.parameters(), **optimizer_kwargs(config))
        update(net, opt, gradients[0]); before = net.weight.detach().clone()
        update(net, opt, gradients[1]); state = snapshot(net, opt, 2)
        dot = float(((net.weight.detach() - before) * torch.tensor([gradients[1]])).double().sum())
        if not (dot > 0 if beta1 else dot < 0): raise AssertionError('sign-reversal control failed')
        validate_saved_optimizer(config, state['optimizer'])
        restored = model()
        with torch.no_grad(): restored.weight.copy_(state['parameters']['weight'].bfloat16().float())
        resumed = torch.optim.AdamW(restored.parameters(), **optimizer_kwargs(config))
        receipt = restore_local_shard(restored, resumed, state, rank=0, world_size=1, updates=2)
        if not equal(state, snapshot(restored, resumed, 2)): raise AssertionError('restore differs')
        update(net, opt, gradients[2]); update(restored, resumed, gradients[2])
        if not equal(snapshot(net, opt, 3), snapshot(restored, resumed, 3)):
            raise AssertionError('resumed trajectory differs')
        rejected = model()
        with torch.no_grad(): rejected.weight.copy_(state['parameters']['weight'].bfloat16().float())
        wrong = dict(config, adam_betas=[0. if beta1 else .9, .999])
        wrong_opt = torch.optim.AdamW(rejected.parameters(), **optimizer_kwargs(wrong))
        untouched = snapshot(rejected, wrong_opt, 0)
        try: restore_local_shard(rejected, wrong_opt, state, rank=0, world_size=1, updates=2)
        except ValueError as exc:
            if 'hyperparameters' not in str(exc): raise
        else: raise AssertionError('changed momentum was accepted')
        if not equal(untouched, snapshot(rejected, wrong_opt, 0)):
            raise AssertionError('failed restore mutated parameters or optimizer')
        reports.append(dict(beta1=beta1, second_step_gradient_dot_delta=dot,
                            exact_resume_and_next_update=True, mismatch_rejected_before_mutation=True,
                            restore=receipt))
    return dict(passed=True, original_default_trajectory_exact=True, reports=reports,
                torch_version=torch.__version__,
                scope='Synthetic CPU FP32 masters/Adam restored from BF16-rounded exports. '
                'No actual FSDP topology, NPU/audio execution or task improvement verified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(exist_ok=False)
    try:
        config = json.loads(args.config.read_text())
        for path, digest in config['pins'].items():
            if file_hash(path) != digest: raise ValueError('probe source changed')
        atomic_json(args.output/'result.json', run())
        atomic_json(args.output/'terminal.json', dict(exit_code=0))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json', dict(exit_code=1, error=repr(exc))); raise

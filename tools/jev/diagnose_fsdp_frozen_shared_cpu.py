#!/usr/bin/env python3
"""Small CPU reproduction; no Qwen, NPU, verl or candidate kernels involved."""
import argparse
import contextlib
import io
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import CustomPolicy


class Model(torch.nn.Module):
    def __init__(self, tied):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.head = torch.nn.Linear(8, 16, bias=False)
        if tied:
            self.head.weight = self.embedding.weight
        self.embedding.weight.requires_grad_(False)
        self.head.weight.requires_grad_(False)
        self.adapter = torch.nn.Linear(8, 8, bias=False)

    def forward(self, ids):
        return self.head(self.adapter(self.embedding(ids)))


def worker(rank, world, tied, root):
    root = Path(root)
    torch.set_num_threads(1)
    torch.manual_seed(317)
    dist.init_process_group('gloo', init_method=(root/'init').resolve().as_uri(), rank=rank, world_size=world)
    record = dict(rank=rank, world_size=world, tied=tied, torch_version=torch.__version__, updates=0)
    try:
        model = Model(tied)
        wrapped = FSDP(model, device_id=torch.device('cpu'), use_orig_params=True,
            auto_wrap_policy=CustomPolicy(lambda m: m is model.adapter))
        opt = torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad], lr=1e-4)
        stream = io.StringIO()
        try:
            with contextlib.redirect_stderr(stream), contextlib.redirect_stdout(stream):
                for step in range(2):
                    opt.zero_grad()
                    wrapped(torch.tensor([1, 2])).square().mean().backward()
                    opt.step()
                    record['updates'] = step + 1
            record['passed'] = True
        except Exception as exc:
            record.update(passed=False, error=repr(exc))
        (root/f'rank-{rank}.log').write_text(stream.getvalue())
        (root/f'rank-{rank}.json').write_text(json.dumps(record, indent=2)+'\n')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for world in (1, 2):
        for tied in (True, False):
            root = args.output/f'dp{world}-tied{int(tied)}'
            root.mkdir()
            mp.spawn(worker, args=(world, tied, str(root)), nprocs=world, join=True)
            results.extend(json.loads(path.read_text()) for path in sorted(root.glob('rank-*.json')))
    result = dict(results=results, scope='CPU FP32 frozen root, trainable child; two Adam updates. Not NPU capacity or production acceptance.')
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

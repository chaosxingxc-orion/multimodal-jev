#!/usr/bin/env python3
"""Restore actual saved shard tensors in the pinned runtime without NPUs."""
import argparse
import json
from pathlib import Path

import torch

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.span_fsdp_resume import restore_local_shard, validate_cursor


def audit(root):
    reports = []
    for rank in range(4):
        path = root / 'run' / f'rank-{rank}' / 'optimizer-local.pt'
        state = torch.load(path, map_location='cpu', weights_only=True)

        class Layout(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.entries = [(n, torch.nn.Parameter(state['parameters'][n].bfloat16().float()))
                                for n in state['names']]

            def named_parameters(self):
                return iter(self.entries)

        model = Layout()
        optimizer = torch.optim.AdamW([p for _, p in model.entries], lr=1e-4,
                                      weight_decay=0., foreach=False, fused=False)
        result = restore_local_shard(model, optimizer, state, rank=rank, world_size=4, updates=1)
        previous = json.loads((path.parent / 'selection.json').read_text())
        current = dict(previous, batch_index=previous['batch_index'] + 4)
        validate_cursor(previous, current, 4)
        reports.append(dict(**result, checkpoint_sha256=file_hash(path),
                            next_pair_index=current['batch_index']))
    return dict(passed=True, ranks=reports, torch_version=torch.__version__,
                scope='Actual saved FSDP master/Adam tensors restored on CPU using reconstructed '
                'local layouts; optimizer metadata unmodified. Not live FSDP restore or NPU update/parity.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    atomic_json(args.output, result)
    print(json.dumps(result))

#!/usr/bin/env python3
"""Bounded, isolated physical-4 native Span SFT pilot."""
import argparse
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json, file_hash

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        import torch
        import torch_npu
        from multimodal_jev.span_training import train
        if sorted(p.name for p in Path('/dev').glob('davinci[0-9]*')) != ['davinci4'] or torch.npu.device_count() != 1:
            raise ValueError('isolated physical NPU 4 required')
        torch.npu.set_device(0)
        config = json.loads(args.config.read_text())
        for path, digest in config['pins'].items():
            if file_hash(path) != digest:
                raise ValueError('pinned source changed: ' + path)
        result = train(config, args.output)
        atomic_json(args.output / 'result.json', result)
        atomic_json(args.output / 'terminal.json', dict(exit_code=0, finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(args.output / 'terminal.json', dict(exit_code=1, error=repr(exc), finished_unix=time.time()))
        raise

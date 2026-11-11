#!/usr/bin/env python3
"""Run a bounded sequence of authenticated single-NPU Span RL updates."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json
from multimodal_jev.span_training_chain import read, run_chain

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        if sorted(p.name for p in Path('/dev').glob('davinci[0-9]*')) != ['davinci4']:
            raise ValueError('isolated physical NPU 4 required')
        run_chain(read(args.config), args.output, Path(__file__).parent)
        atomic_json(args.output/'terminal.json', dict(exit_code=0, finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json', dict(exit_code=1, error=repr(exc), finished_unix=time.time()))
        raise

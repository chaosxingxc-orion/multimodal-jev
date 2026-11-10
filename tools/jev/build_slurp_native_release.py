#!/usr/bin/env python3
"""Export complete native SLURP without discarding repeated entity occurrences."""
import argparse
import fcntl
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json
from multimodal_jev.slurp_native import build

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('root', 'output-root', 'state'): p.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args(); args.state.mkdir(parents=True, exist_ok=True)
    with (args.state/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.state/'result.json').exists(): raise FileExistsError('completed release retained')
        try:
            path, receipt = build(args.root, args.output_root, args.state)
            atomic_json(args.state/'result.json', {'path': str(path), **receipt})
            atomic_json(args.state/'terminal.json', {'exit_code': 0, 'finished_unix': time.time()})
            print(json.dumps({'path': str(path), 'splits': receipt['splits']}, indent=2))
        except BaseException as exc:
            atomic_json(args.state/'terminal.json', {'exit_code': 1, 'error': repr(exc), 'finished_unix': time.time()})
            raise

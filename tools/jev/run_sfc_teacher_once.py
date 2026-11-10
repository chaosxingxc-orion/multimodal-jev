#!/usr/bin/env python3
"""Execute one registered SFC reflection; never retry an ambiguous API attempt."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from multimodal_jev.baseline import atomic_json
from multimodal_jev.sfc_teacher_wire import check_config, private_write, stage_wire


def execute_once(root, intent_hash):
    root = Path(root).resolve()
    cfg, intent, _, _ = check_config(root, intent_hash)
    if (root/'response').exists() or (root/'process-terminal.json').exists():
        raise FileExistsError('prior attempt exists; reconcile without retry')
    # O_EXCL durable intent precedes subprocess creation. Failure after this
    # point remains a recorded attempt even if no PID or response was persisted.
    private_write(root/'process-launch.json', dict(dispatch_intent=True,
        intent_sha256=intent_hash, started_unix=time.time(), automatic_retry=False))
    try:
        with (root/'process.log').open('xb') as log:
            (root/'process.log').chmod(0o600)
            process = subprocess.Popen([sys.executable, intent['executor'], '--config',
                str(root/'executor-config.private.json')], stdout=log, stderr=log)
            atomic_json(root/'process-launch.json', dict(dispatch_intent=True,
                intent_sha256=intent_hash, pid=process.pid, started_unix=time.time(), automatic_retry=False))
            try:
                code = process.wait(timeout=cfg['total_timeout_seconds']+60)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
                raise TimeoutError('teacher process exceeded deadline; remote spend may be unknown')
        private_write(root/'process-terminal.json', dict(exit_code=code,
            intent_sha256=intent_hash, finished_unix=time.time(), automatic_retry=False))
        if code != 0:
            raise RuntimeError('teacher request failed; preserve evidence, no replay')
    except BaseException:
        if not (root/'process-terminal.json').exists():
            private_write(root/'process-terminal.json', dict(exit_code=1,
                intent_sha256=intent_hash, finished_unix=time.time(), automatic_retry=False,
                remote_spend='unknown; inspect private executor records'))
        raise
    return stage_wire(root, root/'candidates', intent_hash)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--intent-hash',required=True)
    a=p.parse_args()
    print(json.dumps(execute_once(a.root,a.intent_hash),indent=2))

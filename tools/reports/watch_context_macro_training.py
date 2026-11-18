"""Schedule the existing exact-handle observer without model-driven polling.

Read-only, one attempt, 900-second intervals shared with manual observations.
Never starts, stops, retries or collects a training job.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import time

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.launch_context_batch_probe_candidate import write_new
from tools.reports.observe_context_macro_training import observe, INTERVAL


def deadline(result, now):
    if result.get('status') == 'observed_exact_attempt':
        state = result.get('containers', {}).get('client', {}).get('state', {})
        if not state.get('Running', True) and state.get('Status') in ('exited', 'dead'):
            return None
    return max(now + 1, result.get('next_ordinary_observation_unix', now + INTERVAL))


def identity(record):
    proc = Path('/proc') / str(record['pid'])
    return ((proc/'stat').read_text().split(') ', 1)[1].split()[19] == record['start_ticks']
            and hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest() == record['cmdline_sha256'])


def run(c, provenance):
    root = Path(c['output']); check_paths({'watcher': root})
    if c['interval_seconds'] != INTERVAL:
        raise ValueError('shared 900-second observation interval required')
    for path, digest in c['pins'].items():
        if file_hash(path) != digest:
            raise ValueError('observer dependency changed: ' + path)
    operations = json.loads(Path(c['operations']).read_text())
    owner = json.loads(Path(c['owner_driver']).read_text())
    fd = os.pidfd_open(owner['pid'])
    try:
        if not identity(owner):
            raise ValueError('exact live owner required')
        root.mkdir(parents=True, exist_ok=False)
        write_new(root/'config.json', c); write_new(root/'provenance.json', provenance)
        proc = Path('/proc') / str(os.getpid())
        write_new(root/'driver.json', dict(pid=os.getpid(),
            start_ticks=(proc/'stat').read_text().split(') ', 1)[1].split()[19],
            cmdline_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest()))
        poll = select.poll(); poll.register(fd, select.POLLIN)
        while True:
            if poll.poll(0):
                # Owner completion is not evidence of remote job success.
                write_new(root/'terminal.json', dict(status='owner_exited_observer_stopped',
                    training_success_verified=False, restart=False))
                return
            try:
                result = observe(operations)
            except Exception as exc:
                result = dict(status='local_observation_error', error=repr(exc),
                    next_ordinary_observation_unix=time.time()+INTERVAL)
            write_new(root/(str(time.time_ns())+'-result.json'), result)
            due = deadline(result, time.time())
            if due is None:
                write_new(root/'terminal.json', dict(status='client_terminal_observer_stopped',
                    training_success_verified=False, restart=False))
                return
            # This wait runs in a detached script; no model wake-ups or remote polling.
            poll.poll(max(1, int((due-time.time())*1000)))
    finally:
        os.close(fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--local', required=True)
    args = parser.parse_args(); config, provenance = load_config(args.config, args.local)
    run(config, provenance)

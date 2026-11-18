"""Detached 900-second listener for one declared replica-training run.

Calls only the deterministic observer. Never starts, stops, retries or collects
the job, and never turns a running state into a success claim.
"""
import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.launch_context_batch_probe_candidate import write_new
from tools.reports.observe_context_replica_training import INTERVAL, handle, observe


def summary(result):
    state = result.get('state') or {}
    progress = result.get('progress') or {}
    updates = progress.get('updates') or {}
    return dict(observed_unix=result.get('observed_unix'), status=result.get('status'),
                container=state.get('Status'), exit_code=state.get('ExitCode'),
                updates=updates,
                policy_commits=progress.get('policy_commits'),
                last_policy_commit=progress.get('last_policy_commit'),
                checkpoint_commits=progress.get('checkpoint_commits'),
                result_present=bool(progress.get('result.json')),
                next_ordinary_observation_unix=result.get('next_ordinary_observation_unix'))


def terminal(result):
    state = result.get('state') or {}
    return (not state.get('Running', True)) and state.get('Status') in ('exited', 'dead')


def watch_root(c):
    """Observation state may sit on another verified local mount than the handle."""
    root = c.get('watch_root')
    return Path(root) if root else Path(c['operations_root']) / (c['tag'] + '-watch')


def run(c, provenance):
    base = Path(c['operations_root'])
    root = watch_root(c)
    check_paths({'watch': root})
    root.mkdir(parents=True, exist_ok=True)
    write_new(root / 'config.json', c) if not (root / 'config.json').exists() else None
    write_new(root / 'provenance.json', provenance) if not (root / 'provenance.json').exists() else None
    owner = (root / 'owner.lock').open('a')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        try:
            result = observe(c)
        except Exception as exc:
            result = dict(status='local_observation_error', error=repr(exc),
                          next_ordinary_observation_unix=time.time() + INTERVAL)
        write_new(root / (str(time.time_ns()) + '-result.json'), result)
        print(json.dumps(summary(result)), flush=True)
        if terminal(result):
            write_new(root / 'terminal.json', dict(
                status='container_terminal_listener_stopped',
                training_success_verified=False,
                exit_code=(result.get('state') or {}).get('ExitCode'),
                collect_and_audit_required=True, restart=False))
            return 0
        due = result.get('next_ordinary_observation_unix') or time.time() + INTERVAL
        time.sleep(max(1, int(due - time.time())))


def detach(template, local):
    c, _ = load_config(template, local)
    if c.get('interval_seconds', INTERVAL) != INTERVAL:
        raise ValueError('shared 900-second observation interval required')
    handle(c)
    root = watch_root(c).with_name(watch_root(c).name + '-driver')
    check_paths({'watch_driver': root})
    root.mkdir(parents=True, exist_ok=False)
    argv = [sys.executable, '-B', '-m', 'tools.reports.watch_context_replica_training',
            '--config', str(Path(template).resolve()), '--local', str(Path(local).resolve())]
    with (root / 'driver.log').open('x') as log:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(pid=child.pid, argv=argv, started_unix=time.time(),
                  process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ', 1)[1].split()[19])
    write_new(root / 'started.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--local', required=True)
    parser.add_argument('--action', choices=['run', 'detach'], default='run')
    args = parser.parse_args()
    cfg, provenance = load_config(args.config, args.local)
    if cfg.get('interval_seconds', INTERVAL) != INTERVAL:
        raise ValueError('shared 900-second observation interval required')
    if args.action == 'detach':
        print(json.dumps(detach(args.config, args.local), indent=2))
    else:
        raise SystemExit(run(cfg, provenance))

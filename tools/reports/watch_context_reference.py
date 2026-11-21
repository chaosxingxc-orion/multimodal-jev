"""Observe original TRAIN scoring as soon as an endpoint successor launches it."""

import argparse
import json
import os
from pathlib import Path
import re
import select
import subprocess
import sys
import time

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.stage2_view_runtime import pinned
from tools.reports.context_replica_training_operations import path as canonical
from tools.reports.observe_context_reference import run as observe
from tools.reports.observe_context_replica_training import INTERVAL, handle


def read_pin(ref):
    return pinned(ref['path'], ref['sha256'])


def inputs(c):
    source = read_pin(c['successor_config'])
    owner = read_pin(c['successor_driver'])
    root = Path(canonical(source['output']))
    chain = source['reference']
    declared = dict(chain['deployment'], **chain['reference_operations'])
    declared.update(package=canonical(chain['reference']['output']),
                    lease_deadline_unix=source['lease']['deadline_unix'],
                    lease_stop_grace_seconds=source['lease']['stop_grace_seconds'])
    base = Path(canonical(declared['operations_root']))
    output = Path(canonical(c['output']))
    if (Path(c['successor_config']['path']) != root/'config.json'
            or Path(c['successor_driver']['path']) != root.with_name(root.name+'-driver')/'started.json'
            or 'tools.reports.context_stage2_endpoint_successor' not in owner['argv']
            or not re.fullmatch(r'[a-z0-9][a-z0-9-]*', declared['tag'])
            or any(output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve())
                   for p in (root, base, Path(declared['package'])))
            or type(c['discovery_seconds']) is not int or not 0 < c['discovery_seconds'] <= INTERVAL):
        raise ValueError('exact successor and fresh independent monitor location required')
    check_paths({'reference_watcher': output})
    return owner, root, output, declared, canonical(chain['reference']['remote_root'])


def read_available(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        # The original lifecycle publishes some receipts directly to fresh files.
        return None


def discover(layout, remote_root, output, binding):
    base = Path(layout['operations_root'])
    stem = base/layout['tag']
    prior = base/'finalization/config.json'
    launch_path = stem.with_name(stem.name+'-launch.json')
    if binding is None:
        if not (prior.is_file() and read_available(launch_path) is not None):
            return None
        pin = dict(path=str(prior), sha256=file_hash(prior))
        config = read_pin(pin)
        if any(config.get(k) != v for k, v in layout.items()):
            raise ValueError('reference lifecycle differs from declared successor')
        _, launch = handle(config)
        if (launch['host'] != layout['host'] or launch['remote_root'] != remote_root
                or config['reference_job']['remote_root'] != remote_root
                or not re.fullmatch('[0-9a-f]{64}', launch['container_id'])):
            raise ValueError('exact reference launch identity required')
        binding = dict(operations=pin, launch_sha256=file_hash(launch_path), container_id=launch['container_id'])
        commit_json(output/'binding.json', binding)
    elif file_hash(launch_path) != binding['launch_sha256']:
        raise ValueError('original scoring launch changed')
    return binding


def current_status(layout, output, binding, *, observe_due):
    if binding is None:
        return None, None
    stem = Path(layout['operations_root'])/layout['tag']
    waited = read_available(stem.with_name(stem.name+'-wait.json'))
    if waited is not None:
        if waited.get('container_id') != binding['container_id'] or type(waited.get('exit_code')) is not int:
            raise ValueError('container wait belongs to a different job')
        return None, dict(status='original_scoring_container_exited', **waited)
    if not observe_due:
        return None, None
    result = observe(dict(operations=binding['operations'], observations_root=str(output/'observations')))
    state = result.get('state', {})
    if state.get('Running') is False and state.get('Status') in ('exited', 'dead'):
        return result, dict(status='original_scoring_container_exited',
                            container_id=binding['container_id'], exit_code=state['ExitCode'])
    return result, None


def run(c, provenance):
    owner, original, output, layout, remote_root = inputs(c)
    output.mkdir(parents=True, exist_ok=False)
    commit_json(output/'config.json', c)
    commit_json(output/'provenance.json', provenance)
    binding, due = None, 0
    fd = None
    poll = select.poll()
    try:
        if not (original/'terminal.json').is_file():
            try:
                fd = os.pidfd_open(owner['pid'])
                proc = Path('/proc')/str(owner['pid'])
                ticks = (proc/'stat').read_text().split(') ', 1)[1].split()[19]
                argv = (proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
                if ticks != owner['process_start_ticks'] or argv != owner['argv']:
                    raise ValueError('successor process identity changed')
                poll.register(fd, select.POLLIN)
            except (ProcessLookupError, FileNotFoundError):
                if not (original/'terminal.json').is_file():
                    raise ValueError('successor disappeared without terminal receipt') from None
        while True:
            original_done = (original/'terminal.json').is_file() or bool(poll.poll(0))
            binding = discover(layout, remote_root, output, binding)
            try:
                result, terminal = current_status(layout, output, binding,
                                                  observe_due=not original_done and time.time() >= due)
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                result = dict(status='observation_failed', error=repr(exc),
                              next_ordinary_observation_unix=time.time() + INTERVAL)
                terminal = None
            if result is not None:
                due = result['next_ordinary_observation_unix']
                print(json.dumps(result), flush=True)
            if terminal is None and original_done:
                if not (original/'terminal.json').is_file():
                    raise ValueError('successor exited without terminal receipt')
                terminal = dict(status='original_successor_stopped', launch_bound=binding is not None)
            if terminal is not None:
                commit_json(output/'terminal.json', dict(terminal, independent_audit_verified=False))
                return
            poll.poll(c['discovery_seconds'] * 1000)
    except Exception as exc:
        commit_json(output/'terminal.json', dict(status='watcher_failed', error=repr(exc)))
        raise
    finally:
        if fd is not None:
            os.close(fd)


def detach(template, local, c):
    _, _, output, _, _ = inputs(c)
    driver = output.with_name(output.name+'-driver')
    if output.exists():
        raise FileExistsError(output)
    check_paths({'reference_watcher_driver': driver})
    driver.mkdir(parents=True, exist_ok=False)
    argv = [sys.executable, '-B', '-m', __spec__.name, '--config', str(Path(template).resolve()),
            '--local', str(Path(local).resolve())]
    with (driver/'driver.log').open('x') as log:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    ticks = Path(f'/proc/{child.pid}/stat').read_text().split(') ', 1)[1].split()[19]
    receipt = dict(pid=child.pid, process_start_ticks=ticks, argv=argv)
    commit_json(driver/'started.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    parser.add_argument('--action', choices=['run', 'detach'], default='run')
    args = parser.parse_args()
    config, provenance = load_config(args.config, args.local)
    if args.action == 'detach':
        print(json.dumps(detach(args.config, args.local, config)))
    else:
        run(config, provenance)

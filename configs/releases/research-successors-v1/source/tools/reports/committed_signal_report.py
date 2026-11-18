"""Configured, local-only TRAIN diagnostics using the existing frozen analyzers.

No model invocation, SSH, scheduling, cleanup, or experiment admission. Run from
the repository root. Incomplete runs are preserved and never silently retried.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from multimodal_jev.local_runtime_storage import check_paths


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    """Durable local commit; caller has checked the output mount."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def verify_sources(sources):
    for path, expected in sources.items():
        if sha(path) != expected:
            raise ValueError('Source changed: ' + str(path))


def load_observation(config):
    proof_path = Path(config['proof'])
    if sha(proof_path) != config['proof_sha256']:
        raise ValueError('Proof identity changed')
    proof = read(proof_path)
    step = config['step']
    if (type(step) is not int or step <= 0 or proof['passed'] is not True
            or proof['step'] != step):
        raise ValueError('Passed proof for the requested step required')
    root = Path(config['collection']).resolve()
    if root != Path(proof['collection']).resolve():
        raise ValueError('Collection differs from audited location')
    latest, sources = [], {str(proof_path): config['proof_sha256']}
    for rank in range(4):
        name = f'run/rank-{rank}/update-{step:04d}.json'
        path = root / name
        expected = proof['files'][name]
        if sha(path) != expected:
            raise ValueError('Committed rank record changed: ' + name)
        data = read(path)
        if data['update'] != step or data['rollout']['update'] != step - 1:
            raise ValueError('Actor/behavior step mismatch')
        sources[str(path)] = expected
        latest.append(dict(path=str(path), data=data))
    if any(row['data']['rollout'] != latest[0]['data']['rollout'] for row in latest[1:]):
        raise ValueError('Replicated rollout mismatch')
    observation = dict(container_id=proof['container_id'], image=proof['image'],
        progress={'latest': latest},
        provenance={'proof': str(proof_path), 'sha256': config['proof_sha256']},
        scope='derived from immutable committed checkpoint; no new remote observation')
    return observation, sources


def compute(observation, config):
    from tools.reports.analyze_context_live_reward_support import run as support
    from tools.reports.replay_context_reward_groups import run as replay
    args = (observation, Path(config['release']), Path(config['schedule']),
            Path(config['receipts']), Path(config['cache_proof']))
    return support(*args), replay(*args)


def run(config_path):
    config_path = Path(config_path).resolve()
    config_hash = sha(config_path)
    config = read(config_path)
    if config['schema'] != 'committed-signal-report-v1':
        raise ValueError('Unsupported configuration schema')
    expected_keys = {'schema', 'step', 'proof', 'proof_sha256', 'collection',
                     'release', 'schedule', 'receipts', 'cache_proof', 'output_directory'}
    if set(config) != expected_keys:
        raise ValueError('Unknown or missing configuration fields')
    output = Path(config['output_directory']).resolve()
    storage = check_paths({'report': output})
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'owner.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publication = output / 'COMPLETE.json'
        if publication.exists():
            manifest = read(publication)
            if manifest['config_sha256'] != config_hash:
                raise ValueError('Existing report belongs to another configuration')
            verify_sources(manifest['sources'])
            verify_sources({str(output / name): digest for name, digest in manifest['outputs'].items()})
            return dict(status='already_complete', manifest=str(publication))
        if (output / 'status.json').exists():
            raise ValueError('Incomplete run preserved; investigate before a versioned rerun')
        started = time.time()
        write(output / 'status.json', dict(phase='running', started_unix=started,
              config_sha256=config_hash, pid=os.getpid()))
        try:
            observation, sources = load_observation(config)
            sources.update({str(config_path): config_hash, str(Path(__file__).resolve()): sha(__file__),
                'src/multimodal_jev/local_runtime_storage.py': sha('src/multimodal_jev/local_runtime_storage.py')})
            write(output / 'observation.json', observation)
            support, groups = compute(output / 'observation.json', config)
            for result in (support, groups):
                if result['actor_update'] != config['step'] or result['behavior_update'] != config['step'] - 1:
                    raise ValueError('Analyzer returned a different update')
                for path, digest in result['sources'].items():
                    if path in sources and sources[path] != digest:
                        raise ValueError('Conflicting source identity: ' + path)
                    sources[path] = digest
            verify_sources(sources)
            write(output / 'reward-support.json', support)
            write(output / 'reward-groups.json', groups)
            rank0 = observation['progress']['latest'][0]['data']
            summary = dict(actor_update=config['step'], behavior_update=config['step'] - 1,
                summaries=support['summaries'], paired_changes=support['paired_exact_changes'],
                groups=groups['summary'],
                runtime={k: rank0[k] for k in ('preparation_seconds', 'actor_seconds', 'total_seconds', 'input_service', 'metrics')},
                scope='Single authenticated TRAIN batch; not selection efficacy, calibration or causal attribution.',
                complex_training_admitted=False)
            write(output / 'summary.json', summary)
            outputs = {name: sha(output / name) for name in
                       ('observation.json', 'reward-support.json', 'reward-groups.json', 'summary.json')}
            verify_sources(sources)
            write(publication, dict(schema='committed-signal-report-publication-v1',
                config_sha256=config_hash, sources=sources, outputs=outputs, storage=storage,
                started_unix=started, completed_unix=time.time(), complex_training_admitted=False))
            write(output / 'status.json', dict(phase='complete', manifest=str(publication)))
            return dict(status='complete', manifest=str(publication))
        except Exception as exc:
            write(output / 'status.json', dict(phase='failed_preserve_evidence',
                config_sha256=config_hash, error=repr(exc), automatic_retry=False))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config)))


if __name__ == '__main__':
    main()

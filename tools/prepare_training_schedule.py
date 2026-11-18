"""Macro-configured wrapper for the admitted full-population schedule builder.

The v1 profile deliberately rejects unsupported knobs instead of silently
ignoring them. No sampling of selection/test and no model or remote calls.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path

from multimodal_jev.local_config import load_config
from multimodal_jev.context_training_schedule import load_schedule
from multimodal_jev.slurp_official_training import load
from tools.jev.prepare_context_training_schedule import prepare
from tools.reports.committed_signal_report import check_paths, read, sha, verify_sources, write


PROFILE = dict(seed=317, epochs=2, global_batch=128, world_size=4, microbatch=8,
               split='learning', coverage='full_without_replacement',
               class_order='class_spread', length_grouping='within_global_batch',
               padding_weight=0, oversampling=False)
DEPENDENCIES = (
    'tools/prepare_training_schedule.py',
    'src/multimodal_jev/local_config.py',
    'src/multimodal_jev/local_runtime_storage.py',
    'tools/reports/committed_signal_report.py',
    'tools/jev/prepare_context_training_schedule.py',
    'src/multimodal_jev/context_training_schedule.py',
    'src/multimodal_jev/span_class_spread.py',
    'src/multimodal_jev/span_length_groups.py',
    'src/multimodal_jev/slurp_official_training.py',
    'src/multimodal_jev/context_candidate_task.py',
)


def validate(config):
    if (set(config) != {'schema', 'release', 'release_sha256', 'ledger', 'ledger_sha256',
                       'profile', 'output_directory'}
            or config['schema'] != 'configured-context-training-schedule-v1'):
        raise ValueError('Unsupported configuration schema or fields')
    # Compare JSON types too: booleans are not valid substitutes for integers.
    if json.dumps(config['profile'], sort_keys=True) != json.dumps(PROFILE, sort_keys=True):
        raise ValueError('Unsupported schedule profile; v1 fixes seed317/two epochs/global128/DP4/micro8')


def audit(schedule, release, release_sha):
    rows, _, _ = load(release, release_sha, 'learning', purpose='optimize')
    batches, manifest = load_schedule(schedule, sha(schedule / 'MANIFEST.json'),
        release_sha256=release_sha, population={r['id']: r for r in rows})
    return dict(recordings_per_epoch=len(rows), updates=len(batches),
        real_exposures=sum(not s['padding'] for b in batches for s in b['slots']),
        padding_slots=sum(s['padding'] for b in batches for s in b['slots']),
        training_dispatch_allowed=manifest['training_dispatch_allowed'])


def run(template, local):
    config, provenance = load_config(template, local)
    validate(config)
    output = Path(config['output_directory']).resolve()
    check_paths({'schedule': output})
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'owner.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        complete = output / 'COMPLETE.json'
        if complete.exists():
            old = read(complete)
            if old['provenance'] != provenance:
                raise ValueError('Existing schedule configuration changed; use a new version')
            verify_sources(old['sources'])
            verify_sources({str(output / name): digest for name, digest in old['outputs'].items()})
            return dict(status='already_complete', **old['summary'])
        if (output / 'status.json').exists():
            raise ValueError('Incomplete schedule preserved; no automatic retry')
        write(output / 'status.json', dict(phase='building', provenance=provenance))
        try:
            release, ledger = Path(config['release']), Path(config['ledger'])
            sources = {str(Path(p).resolve()): sha(p) for p in DEPENDENCIES}
            sources[str(Path(template).resolve())] = provenance['template_sha256']
            sources[str(Path(local).resolve())] = provenance['local_bindings_sha256']
            sources[str(release / 'MANIFEST.json')] = config['release_sha256']
            sources[str(ledger)] = config['ledger_sha256']
            verify_sources(sources)
            # Bind the complete release files which the admitted loader checks.
            manifest = read(release / 'MANIFEST.json')
            for name, digest in manifest['files'].items():
                path = release / name
                if not path.resolve().is_relative_to(release.resolve()):
                    raise ValueError('Release entry escapes root')
                sources[str(path)] = digest
            verify_sources(sources)
            target = output / 'schedule'
            prepare(release, config['release_sha256'], ledger, config['ledger_sha256'], target)
            summary = audit(target, release, config['release_sha256'])
            if summary['training_dispatch_allowed'] is not False:
                raise ValueError('Schedule preparation must not admit training')
            # Existing builder writes JSON; durable publication uses shared writer.
            for name in ('epoch-0.json', 'epoch-1.json', 'MANIFEST.json'):
                with (target / name).open('rb') as stream:
                    os.fsync(stream.fileno())
            fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            verify_sources(sources)
            outputs = {f'schedule/{name}': sha(target / name) for name in
                       ('epoch-0.json', 'epoch-1.json', 'MANIFEST.json')}
            write(complete, dict(schema='configured-training-schedule-publication-v1',
                provenance=provenance, sources=sources, outputs=outputs,
                summary=summary, training_dispatch_allowed=False))
            write(output / 'status.json', dict(phase='complete', **summary))
            return dict(status='complete', **summary)
        except Exception as exc:
            write(output / 'status.json', dict(phase='failed_preserve_evidence',
                error=repr(exc), automatic_retry=False))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', default=os.environ.get('JEV_LOCAL_CONFIG', '.local/config/macros.json'))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.local)))


if __name__ == '__main__':
    main()

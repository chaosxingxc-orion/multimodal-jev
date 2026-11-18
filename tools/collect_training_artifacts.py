"""Configured terminal training collection; reuse the existing transport reader.

Adopt mode authenticates an existing archive without contacting a remote host.
Collect mode uses an exact container and never starts/restarts training.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re

from multimodal_jev.local_config import load_config
from tools.jev.collect_context_rl_training import collect, sha
from tools.reports.committed_signal_report import check_paths, read, write


def verify_files(files):
    for name, digest in files.items():
        if sha(name) != digest:
            raise ValueError('Artifact identity changed: ' + name)


def authenticate(launch, collection):
    collection = Path(collection).resolve()
    receipt, archive = collection.with_suffix('.transport.json'), collection.with_suffix('.tar')
    value = read(receipt)
    if (value.get('status') != 'transport_complete_independent_audit_required'
            or value.get('state', {}).get('Status') != 'exited'
            or any(value.get(k) for k in ('bad_pins', 'bad_package', 'launch_binding_mismatches'))
            or any(value.get(k) != launch[k] for k in ('container_id', 'image', 'remote_root'))):
        raise ValueError('Complete matching terminal transport required')
    files = value.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Nonempty collected file manifest required')
    bindings = {'config.json': 'config_sha256', 'service-config.json': 'service_config_sha256',
                'PREPARED.json': 'manifest_sha256'}
    if any(files.get(name) != launch[key] for name, key in bindings.items()):
        raise ValueError('Collected package differs from launch')
    bound = {str(receipt): sha(receipt), str(archive): value['archive_sha256']}
    for name, digest in files.items():
        relative = Path(name)
        path = collection / relative
        if (relative.is_absolute() or '..' in relative.parts or path.resolve() != path
                or not path.is_file() or not re.fullmatch(r'[0-9a-f]{64}', digest)):
            raise ValueError('Unsafe or malformed collected file')
        bound[str(path)] = digest
    verify_files(bound)
    return value, bound


def run(template, local):
    config, provenance = load_config(template, local)
    if (set(config) != {'schema', 'mode', 'host', 'launch', 'launch_sha256', 'collection',
                       'output_directory', 'lock_path'}
            or config['schema'] != 'configured-training-collection-v1'
            or config['mode'] not in ('adopt', 'collect')
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.@-]*', config['host'])):
        raise ValueError('Explicit collection mode and SSH alias required')
    launch = read(config['launch'])
    if (not re.fullmatch(r'[0-9a-f]{64}', launch['container_id'])
            or not re.fullmatch(r'sha256:[0-9a-f]{64}', launch['image'])
            or not Path(launch['remote_root']).is_absolute()):
        raise ValueError('Exact container, image and absolute remote root required')
    root = Path(config['output_directory']).resolve()
    collection = Path(config['collection']).resolve()
    lock = Path(config['lock_path']).resolve()
    if (root == collection or root.is_relative_to(collection) or collection.is_relative_to(root)
            or lock.is_relative_to(collection)):
        raise ValueError('Publication, lock and collected artifacts must be separate')
    sources = {str(Path(template).resolve()): provenance['template_sha256'],
               str(Path(local).resolve()): provenance['local_bindings_sha256'],
               str(Path(config['launch']).resolve()): config['launch_sha256']}
    for name in (__file__, 'tools/jev/collect_context_rl_training.py',
                 'tools/jev/collect_context_rl_selection.py', 'src/multimodal_jev/local_config.py',
                 'src/multimodal_jev/local_runtime_storage.py', 'tools/reports/committed_signal_report.py'):
        sources[str(Path(name).resolve())] = sha(name)
    if any(Path(p).is_relative_to(root) or Path(p).is_relative_to(collection) for p in sources):
        raise ValueError('Inputs must be separate from outputs')
    verify_files(sources)
    paths = {'publication': root, 'lock': lock.parent}
    if config['mode'] == 'collect': paths['collection'] = collection
    check_paths(paths)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(parents=True, exist_ok=True)
        with (root / 'owner.lock').open('a') as publication_owner:
            fcntl.flock(publication_owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            complete = root / 'COMPLETE.json'
            if complete.exists():
                previous = read(complete)
                if previous['provenance'] != provenance:
                    raise ValueError('Configuration changed; use a new version')
                verify_files(previous['sources'])
                verify_files(previous['artifacts'])
                return dict(status='already_complete', **previous['summary'])
            if (root / 'status.json').exists():
                raise ValueError('Partial collection preserved; automatic retry disabled')
            write(root / 'status.json', dict(phase='collecting', mode=config['mode']))
            try:
                if config['mode'] == 'collect':
                    if any(p.exists() for p in (collection, collection.with_suffix('.tar'),
                                               collection.with_suffix('.transport.json'))):
                        raise ValueError('Existing collection; adopt only after original owner finishes')
                    collection.parent.mkdir(parents=True, exist_ok=True)
                    collect(launch, collection, host=config['host'])
                transport, artifacts = authenticate(launch, collection)
                verify_files(sources)
                summary = dict(mode=config['mode'], container_id=launch['container_id'],
                    collected_files=len(transport['files']), exit_code=transport['state']['ExitCode'],
                    collection=str(collection), transport=str(collection.with_suffix('.transport.json')),
                    training_success_certified=False)
                write(complete, dict(schema='configured-training-collection-publication-v1',
                    provenance=provenance, sources=sources, artifacts=artifacts, summary=summary))
                write(root / 'status.json', dict(phase='complete'))
                return dict(status='complete', **summary)
            except Exception as exc:
                write(root / 'status.json', dict(phase='failed_preserve_evidence', error=repr(exc)))
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', default=os.environ.get('JEV_LOCAL_CONFIG', '.local/config/macros.json'))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.local)))


if __name__ == '__main__':
    main()

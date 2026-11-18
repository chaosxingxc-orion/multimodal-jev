"""Publish a retention inventory from a producer's authenticated closure ledger.

No scanning, process discovery, deletion, or model analysis. All producers and
consumers must register jobs/dependencies and cooperate on the experiment lock.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil

from multimodal_jev.local_config import load_config
from tools.collect_training_artifacts import sha, verify_files
from tools.reports.committed_signal_report import check_paths, read, write
from tools.retain_experiment_artifacts import prepare


def authenticate(config):
    if (set(config) != {'schema', 'root', 'ledger', 'ledger_sha256', 'output_directory', 'lock_path'}
            or config['schema'] != 'retention-inventory-publication-config-v1'):
        raise ValueError('Explicit inventory publication configuration required')
    root = Path(config['root']).absolute()
    if (root.resolve() != root or not root.is_dir() or len(root.parts) < 4
            or any(p in {'models', 'datasets', '.git'} for p in root.parts)):
        raise ValueError('Narrow nonsymlink experiment root required')
    ledger = Path(config['ledger']).resolve()
    sources = {str(ledger): config['ledger_sha256']}
    verify_files(sources)
    data = read(ledger)
    required = {'schema', 'root', 'closed', 'dependencies_registered', 'terminal',
                'terminal_sha256', 'jobs', 'required_jobs', 'audits', 'required_audits',
                'files', 'selected_checkpoint_steps', 'pin_manifests', 'protected_paths'}
    if (set(data) != required or data['schema'] != 'experiment-closure-ledger-v1'
            or data['root'] != str(root) or data['closed'] is not True
            or data['dependencies_registered'] is not True):
        raise ValueError('Closed producer ledger and registered dependencies required')

    def bound(path, digest):
        path = Path(path).absolute()
        if path.resolve() != path or not path.is_file():
            raise ValueError('Nonsymlink evidence file required')
        name = str(path)
        if name in sources and sources[name] != digest:
            raise ValueError('Conflicting evidence identity')
        sources[name] = digest
        verify_files({name: digest})
        return path

    def population(kind):
        names = data['required_' + kind]
        rows = data[kind]
        if (not isinstance(names, list) or not names or len(set(names)) != len(names)
                or not all(isinstance(n, str) and n for n in names)
                or not isinstance(rows, dict) or set(rows) != set(names)):
            raise ValueError('Complete declared ' + kind + ' population required')
        return rows

    handles = set()
    for row in population('jobs').values():
        if set(row) != {'receipt', 'sha256', 'container_id'}:
            raise ValueError('Exact terminal job binding required')
        value = read(bound(row['receipt'], row['sha256']))
        state = value.get('state', {})
        if (value.get('container_id') != row['container_id'] or not row['container_id']
                or row['container_id'] in handles or state.get('Status') != 'exited'
                or type(state.get('ExitCode')) is not int or state['ExitCode'] != 0
                or state.get('OOMKilled') is not False
                or state.get('Running', False) is not False):
            raise ValueError('Distinct successful exact terminal jobs required')
        handles.add(row['container_id'])
    for row in population('audits').values():
        if set(row) != {'path', 'sha256'}:
            raise ValueError('Explicit audit binding required')
        if read(bound(row['path'], row['sha256'])).get('passed') is not True:
            raise ValueError('Passed registered audit required')
    terminal_name = Path(data['terminal'])
    if terminal_name.is_absolute() or '..' in terminal_name.parts:
        raise ValueError('Experiment-relative terminal required')
    terminal = bound(root / terminal_name, data['terminal_sha256'])
    value = read(terminal)
    if type(value.get('exit_code')) is not int or value['exit_code'] != 0:
        raise ValueError('Successful experiment terminal required')
    if not isinstance(data['pin_manifests'], dict) or not isinstance(data['protected_paths'], list):
        raise ValueError('Registered dependency pins and protection required')
    for path, digest in data['pin_manifests'].items():
        bound(path, digest)
    # Role/step validity, archives, complete checkpoint protection and all file
    # hashes are checked by the shared retention planner before publication.
    if not isinstance(data['files'], list) or not data['files']:
        raise ValueError('Nonempty producer file ledger required')
    for row in data['files']:
        name = Path(row['path'])
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('Experiment-relative artifact required')
        bound(root / name, row['sha256'])
    return data, sources


def run(template, local):
    config, provenance = load_config(template, local)
    root = Path(config['root']).resolve()
    output, lock = [Path(config[k]).absolute() for k in ('output_directory', 'lock_path')]
    if (output.resolve() != output or lock.resolve() != lock
            or output.is_relative_to(root) or root.is_relative_to(output)
            or lock.is_relative_to(root)):
        raise ValueError('Separate nonsymlink publication and lock required')
    check_paths({'publication': output, 'lock': lock.parent})
    lock.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as owner, (output/'owner.lock').open('a') as pub_owner:
        for stream in (owner, pub_owner):
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        complete = output/'COMPLETE.json'
        if complete.exists():
            old = read(complete)
            if old['provenance'] != provenance:
                raise ValueError('Changed inventory publication; use a new version')
            verify_files(old['sources']); verify_files(old['outputs'])
            return dict(status='already_complete', **old['summary'])
        if (output/'status.json').exists():
            raise ValueError('Partial publication preserved; automatic retry disabled')
        write(output/'status.json', dict(phase='authenticating'))
        try:
            data, sources = authenticate(config)
            sources.update({str(Path(template).resolve()): provenance['template_sha256'],
                            str(Path(local).resolve()): provenance['local_bindings_sha256']})
            for name in (__file__, 'tools/retain_experiment_artifacts.py', 'tools/cache_cleanup.py',
                         'tools/collect_training_artifacts.py', 'tools/jev/collect_context_rl_training.py',
                         'tools/reports/committed_signal_report.py', 'src/multimodal_jev/local_config.py',
                         'src/multimodal_jev/local_runtime_storage.py'):
                sources[str(Path(name).resolve())] = sha(name)
            if any(Path(p).is_relative_to(output) for p in sources):
                raise ValueError('Publication cannot contain source inputs')
            archives, outputs = {}, {}
            archive_root = output/'conclusions'; archive_root.mkdir()
            for i, row in enumerate(data['files']):
                if row['role'] != 'conclusion': continue
                dest = archive_root/f'{i:06d}.archive'
                with (root/row['path']).open('rb') as src, dest.open('xb') as dst:
                    shutil.copyfileobj(src, dst, length=1024*1024)
                    dst.flush(); os.fsync(dst.fileno())
                outputs[str(dest)] = row['sha256']
                archives[row['path']] = str(dest)
            fd = os.open(archive_root, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
            verify_files(outputs); verify_files(sources)
            inventory = {k: data[k] for k in ('terminal', 'terminal_sha256', 'files', 'selected_checkpoint_steps')}
            inventory.update(schema='closed-experiment-inventory-v1', closed=True,
                             evaluation_complete=True, conclusion_archives=archives)
            path = output/'inventory.json'; write(path, inventory)
            outputs[str(path)] = sha(path)
            consumer = dict(schema='closed-experiment-retention-v1', root=str(root), inventory=str(path),
                inventory_sha256=sha(path), pin_manifests=list(data['pin_manifests']),
                protected_paths=data['protected_paths'], output_directory=str(output/'retention-plan'),
                lock_path=str(lock), max_files=max(1,len(data['files'])), max_bytes=2**63-1)
            plan, _ = prepare(consumer)
            # Copy the dependency bindings for downstream macro injection; this
            # is not a delete command and does not modify a team's global macros.
            bindings = {k: consumer[k] for k in ('root','inventory','inventory_sha256',
                        'pin_manifests','protected_paths','lock_path')}
            write(output/'retention-inputs.json', bindings)
            outputs[str(output/'retention-inputs.json')] = sha(output/'retention-inputs.json')
            verify_files(sources); verify_files(outputs)
            summary = dict(inventory=str(path), inventory_sha256=sha(path), files=len(data['files']),
                archived_conclusions=len(archives), candidates=len(plan['files']),
                kept_checkpoint_steps=plan['kept_checkpoint_steps'], deleted_files=0)
            write(complete, dict(schema='retention-inventory-publication-v1', provenance=provenance,
                                 sources=sources, outputs=outputs, summary=summary))
            write(output/'status.json', dict(phase='complete'))
            return dict(status='complete', **summary)
        except Exception as exc:
            write(output/'status.json', dict(phase='failed_preserve_evidence', error=repr(exc)))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', default=os.environ.get('JEV_LOCAL_CONFIG', '.local/config/macros.json'))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.local)))

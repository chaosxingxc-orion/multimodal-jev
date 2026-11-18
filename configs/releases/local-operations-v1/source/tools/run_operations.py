"""Single-process, configured local operations DAG. No shell or remote dispatch.

Failures preserve evidence and never automatically retry a possibly completed
step. Existing tool-specific locks and validations remain authoritative.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re

from multimodal_jev.local_config import load_config
from tools.reports.committed_signal_report import check_paths, read, sha, verify_sources, write


KINDS = {'materialize', 'training_schedule', 'cache_plan', 'signal_report', 'probability_errors', 'training_trajectory'}
MODULES = {
    'materialize': 'tools/configure.py',
    'training_schedule': 'tools/prepare_training_schedule.py',
    'cache_plan': 'tools/cache_cleanup.py',
    'signal_report': 'tools/reports/committed_signal_report.py',
    'probability_errors': 'tools/reports/probability_error_report.py',
    'training_trajectory': 'tools/export_training_trajectory.py',
}

PUBLICATIONS = {
    'training_schedule': ('configured-training-schedule-publication-v1',
                          {'schedule/epoch-0.json', 'schedule/epoch-1.json', 'schedule/MANIFEST.json'}),
    'signal_report': ('committed-signal-report-publication-v1',
                      {'observation.json', 'reward-support.json', 'reward-groups.json', 'summary.json'}),
    'probability_errors': ('probability-error-report-publication-v1', {'probability-errors.json'}),
    'training_trajectory': ('recorded-training-trajectory-publication-v1', {'trajectory.jsonl', 'updates.json'}),
}


def verified_publication(kind, complete):
    """Verify child claims against disk; never adopt current hashes as proof."""
    complete = Path(complete).resolve()
    manifest = read(complete)
    if kind == 'cache_plan':
        if manifest.get('apply') is not False or manifest['summary']['deleted_files'] != 0:
            raise ValueError('Expected a plan-only cache result')
        outputs = {'plan.json': manifest['plan_sha256']}
        sources = read(complete.parent / 'plan.json')['pin_manifests']
    else:
        schema, names = PUBLICATIONS[kind]
        if manifest.get('schema') != schema or set(manifest['outputs']) != names:
            raise ValueError('Incomplete or wrong child publication schema')
        outputs, sources = manifest['outputs'], manifest['sources']
    verify_sources(sources)
    artifacts = {str(complete): sha(complete)}
    for name, digest in outputs.items():
        path = complete.parent / name
        if not path.resolve().is_relative_to(complete.parent):
            raise ValueError('Publication output escapes operation directory')
        artifacts[str(path)] = digest
    verify_sources(artifacts)
    return sources, artifacts


def ordered_steps(config):
    if (set(config) != {'schema', 'output_directory', 'steps'}
            or config['schema'] not in {'local-operations-v1', 'local-operations-v2'}
            or not isinstance(config['steps'], list) or not config['steps']):
        raise ValueError('Nonempty local-operations-v1 configuration required')
    steps = {}
    for step in config['steps']:
        required = {'id', 'kind', 'config', 'depends_on'}
        if step.get('kind') == 'materialize':
            required.add('output')
        if 'signal_from' in step:
            if config['schema'] != 'local-operations-v2' or step.get('kind') != 'probability_errors':
                raise ValueError('signal_from requires a v2 probability_errors step')
            required.add('signal_from')
        if (set(step) != required or not isinstance(step['id'], str)
                or not re.fullmatch(r'[a-z][a-z0-9_-]*', step['id'])
                or step['id'] in steps or step['kind'] not in KINDS
                or not isinstance(step['depends_on'], list)
                or any(not isinstance(d, str) for d in step['depends_on'])
                or len(set(step['depends_on'])) != len(step['depends_on'])):
            raise ValueError('Invalid or duplicate operation step')
        steps[step['id']] = step
    if any(d not in steps for s in steps.values() for d in s['depends_on']):
        raise ValueError('Missing dependency')
    for step in steps.values():
        if 'signal_from' in step:
            upstream = step['signal_from']
            if (not isinstance(upstream, str) or upstream not in step['depends_on']
                    or steps[upstream]['kind'] != 'signal_report'):
                raise ValueError('signal_from must name a direct signal_report dependency')
    ordered, done = [], set()
    while len(done) < len(steps):
        ready = [s for s in steps.values() if s['id'] not in done and set(s['depends_on']) <= done]
        if not ready:
            raise ValueError('Operation dependency cycle')
        for step in ready:
            ordered.append(step)
            done.add(step['id'])
    return ordered


def child_bindings(step, base, publications=None):
    """Only a declared signal publication can supply these two runtime macros."""
    bindings = read(base)
    if 'signal_from' in step:
        if publications is None:
            # Placeholders for preflight: no side effects or inferred artifact paths.
            path, digest = '/pending-signal-publication/COMPLETE.json', '0' * 64
        else:
            path, digest = publications[step['signal_from']]
            if sha(path) != digest:
                raise ValueError('Upstream publication changed before binding')
        bindings['macros'].update(SIGNAL_PUBLICATION=str(path), SIGNAL_PUBLICATION_SHA256=digest)
    return bindings


def execute(step, local, root):
    kind = step['kind']
    if kind == 'materialize':
        from tools.configure import materialize
        result = materialize(step['config'], local, step['output'])
        output = Path(step['output']).resolve()
        return result, [output, output.with_name(output.name + '.provenance.json')]
    if kind == 'training_schedule':
        from tools.prepare_training_schedule import run
        result = run(step['config'], local)
    elif kind == 'cache_plan':
        from tools.cache_cleanup import run
        result = run(step['config'], local, apply=False)
    elif kind == 'training_trajectory':
        from tools.export_training_trajectory import run
        result = run(step['config'], local)
    else:
        from tools.configure import materialize
        resolved = root / 'resolved' / (step['id'] + '.json')
        materialize(step['config'], local, resolved)
        if kind == 'signal_report':
            from tools.reports.committed_signal_report import run
        elif kind == 'probability_errors':
            from tools.reports.probability_error_report import run
        else:
            raise ValueError('Unregistered operation')
        result = run(resolved)
    config, _ = load_config(step['config'], local)
    output = Path(config['output_directory']).resolve()
    complete = output / 'COMPLETE.json'
    _, artifacts = verified_publication(kind, complete)
    return result, [Path(p) for p in artifacts]


def run(template, local):
    config, provenance = load_config(template, local)
    steps = ordered_steps(config)
    root = Path(config['output_directory']).resolve()
    # Capture the used templates and direct orchestration dependencies before writes.
    sources = {str(Path(p).resolve()): sha(p) for p in
               [__file__, 'src/multimodal_jev/local_config.py',
                'src/multimodal_jev/local_runtime_storage.py',
                'tools/reports/committed_signal_report.py', 'tools/configure.py']
               + [MODULES[s['kind']] for s in steps] + [s['config'] for s in steps]}
    sources[str(Path(template).resolve())] = provenance['template_sha256']
    sources[str(Path(local).resolve())] = provenance['local_bindings_sha256']
    verify_sources(sources)
    # Validate all child macros before starting the first side effect.
    for step in steps:
        if 'signal_from' in step:
            from multimodal_jev.local_config import resolve
            resolve(read(step['config']), child_bindings(step, local)['macros'])
        else:
            load_config(step['config'], local)
        if step['kind'] == 'materialize' and Path(step['output']).resolve() in map(Path, sources):
            raise ValueError('Materialized output would overwrite an input')
    check_paths({'operations': root})
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'owner.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        complete = root / 'COMPLETE.json'
        if complete.exists():
            old = read(complete)
            if old['provenance'] != provenance:
                raise ValueError('Workflow configuration changed; use a new version')
            verify_sources(old['sources'])
            verify_sources(old['artifacts'])
            return dict(status='already_complete', completed=old['completed'])
        if (root / 'status.json').exists():
            raise ValueError('Partial workflow preserved; automatic retry disabled')
        completed, artifacts, publications = [], {}, {}
        write(root / 'status.json', dict(phase='starting', provenance=provenance))
        try:
            # Freeze bindings for this workflow; later global edits cannot silently
            # change one child while another uses old values.
            snapshot = root / 'macros.snapshot.json'
            write(snapshot, read(local))
            snapshot_hash = sha(snapshot)
            for step in steps:
                verify_sources(sources)
                verify_sources(artifacts)
                if sha(snapshot) != snapshot_hash:
                    raise ValueError('Workflow macro snapshot changed')
                write(root / 'status.json', dict(phase='executing', step=step['id'], completed=completed))
                binding = snapshot
                if 'signal_from' in step:
                    binding = root / (step['id'] + '.macros.json')
                    write(binding, child_bindings(step, snapshot, publications))
                    artifacts[str(binding)] = sha(binding)
                result, paths = execute(step, binding, root)
                if result.get('status') not in {'complete', 'already_complete', 'resolved', 'already_resolved'}:
                    raise ValueError('Operation did not report completion')
                verify_sources(sources)
                # Retain each child's full declared input dependency set so later
                # workflow reuse cannot ignore changes to training/trajectory data.
                if step['kind'] != 'materialize':
                    for path in paths:
                        if Path(path).name != 'COMPLETE.json':
                            continue
                        child_sources, child_artifacts = verified_publication(step['kind'], path)
                        for name, digest in child_sources.items():
                            if name in sources and sources[name] != digest:
                                raise ValueError('Conflicting child source identity')
                            sources[name] = digest
                        artifacts.update(child_artifacts)
                for path in paths:
                    name, digest = str(Path(path).resolve()), sha(path)
                    if name in artifacts and artifacts[name] != digest:
                        raise ValueError('Artifact changed after publication validation')
                    artifacts[name] = digest
                if step['kind'] == 'signal_report':
                    publication = Path(result['manifest']).resolve()
                    if str(publication) not in artifacts:
                        raise ValueError('Signal publication absent from verified artifacts')
                    manifest = read(publication)
                    if manifest['schema'] != 'committed-signal-report-publication-v1':
                        raise ValueError('Unexpected signal publication schema')
                    verify_sources(manifest['sources'])
                    verify_sources({str(publication.parent / n): h for n, h in manifest['outputs'].items()})
                    publications[step['id']] = (str(publication), sha(publication))
                receipt = root / (step['id'] + '.result.json')
                write(receipt, result)
                artifacts[str(receipt)] = sha(receipt)
                completed.append(step['id'])
            artifacts[str(snapshot)] = snapshot_hash
            verify_sources(sources)
            verify_sources(artifacts)
            write(complete, dict(schema='local-operations-publication-v1', provenance=provenance,
                sources=sources, artifacts=artifacts, completed=completed, training_dispatched=False))
            write(root / 'status.json', dict(phase='complete', completed=completed))
            return dict(status='complete', completed=completed)
        except Exception as exc:
            write(root / 'status.json', dict(phase='failed_preserve_evidence', error=repr(exc),
                completed=completed, automatic_retry=False))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', default=os.environ.get('JEV_LOCAL_CONFIG', '.local/config/macros.json'))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.local)))


if __name__ == '__main__':
    main()

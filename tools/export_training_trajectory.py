"""Export deduplicated recorded rollouts from an existing terminal collection.

No remote collection, model call, action resampling or reward reconstruction.
Action receipts contain seeds and tensor hashes, not the full sampled bit arrays.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re

from multimodal_jev.local_config import load_config
from tools.reports.committed_signal_report import check_paths, read, verify_sources, write
from tools.jev.collect_context_rl_training import sha


def updates(collection, transport):
    """Authenticate every recorded rank update; count replicated queries once."""
    if (transport['status'] != 'transport_complete_independent_audit_required'
            or transport['state']['Status'] != 'exited'
            or any(transport.get(k) for k in ('bad_pins', 'bad_package', 'launch_binding_mismatches'))):
        raise ValueError('Complete terminal transport without identity mismatch required')
    groups = {}
    for name, digest in transport['files'].items():
        match = re.fullmatch(r'run/rank-(\d+)/update-(\d+)\.json', name)
        if match:
            rank, step = map(int, match.groups())
            groups.setdefault(step, {})[rank] = (name, digest)
    if not groups:
        raise ValueError('No recorded training updates')
    config_path = collection / 'config.json'
    if sha(config_path) != transport['files']['config.json']:
        raise ValueError('Collected configuration changed')
    config = read(config_path)
    if len(config['physical_devices']) != 4:
        raise ValueError('Trajectory export v1 requires DP4')
    for step in sorted(groups):
        group = groups[step]
        if set(group) != set(range(4)):
            raise ValueError('Incomplete rank population')
        records, sources = [], {}
        for rank in range(4):
            name, digest = group[rank]
            path = collection / name
            if path.is_symlink() or not path.resolve().is_relative_to(collection.resolve()) or sha(path) != digest:
                raise ValueError('Collected update changed or escaped root')
            value = read(path)
            if value['update'] != step or value['rollout']['update'] != step - 1:
                raise ValueError('Actor/behavior update mismatch')
            records.append(value)
            sources[str(path)] = digest
        rollout = records[0]['rollout']
        if any(r['rollout'] != rollout for r in records[1:]):
            raise ValueError('Rank replicas disagree')
        queries = rollout['queries']
        if (len(queries) != config['global_batch']
                or len({q['request_id'] for q in queries}) != len(queries)):
            raise ValueError('Complete distinct global query batch required')
        for q in queries:
            if (q['behavior_record']['policy_id'] != rollout['behavior_policy_id']
                    or q['reference_record']['policy_id'] != rollout['fixed_reference_policy_id']):
                raise ValueError('Query policy binding mismatch')
        yield step, rollout, records, sources


def run(template, local):
    config, provenance = load_config(template, local)
    if (set(config) != {'schema', 'collection', 'transport', 'transport_sha256', 'output_directory'}
            or config['schema'] != 'recorded-training-trajectory-v1'):
        raise ValueError('Unsupported trajectory configuration')
    collection = Path(config['collection']).resolve()
    receipt = Path(config['transport']).resolve()
    output = Path(config['output_directory']).resolve()
    if output == collection or output.is_relative_to(collection):
        raise ValueError('Export must not modify collected inputs')
    check_paths({'trajectory': output})
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'owner.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        complete = output / 'COMPLETE.json'
        if complete.exists():
            previous = read(complete)
            if previous['provenance'] != provenance:
                raise ValueError('Existing export has another configuration')
            verify_sources(previous['sources'])
            verify_sources({str(output / p): h for p, h in previous['outputs'].items()})
            return dict(status='already_complete', **previous['summary'])
        if (output / 'status.json').exists():
            raise ValueError('Partial export preserved; no automatic retry')
        write(output / 'status.json', dict(phase='exporting', provenance=provenance))
        try:
            sources = {str(receipt): config['transport_sha256'],
                       str(Path(template).resolve()): provenance['template_sha256'],
                       str(Path(local).resolve()): provenance['local_bindings_sha256']}
            for p in (__file__, 'src/multimodal_jev/local_config.py',
                      'src/multimodal_jev/local_runtime_storage.py',
                      'tools/reports/committed_signal_report.py', 'tools/jev/collect_context_rl_training.py'):
                sources[str(Path(p).resolve())] = sha(p)
            verify_sources(sources)
            transport = read(receipt)
            sources[str(collection / 'config.json')] = transport['files']['config.json']
            metadata, count = [], 0
            path = output / 'trajectory.jsonl'
            with path.open('x') as stream:
                for step, rollout, records, bound in updates(collection, transport):
                    sources.update(bound)
                    for q in rollout['queries']:
                        stream.write(json.dumps(dict(actor_update=step, behavior_update=step-1,
                            request_id=q['request_id'], recorded_query=q), allow_nan=False) + '\n')
                        count += 1
                    metadata.append(dict(actor_update=step, behavior_update=step-1,
                        queries=len(rollout['queries']), behavior_policy_id=rollout['behavior_policy_id'],
                        reference_policy_id=rollout['fixed_reference_policy_id'],
                        ranks=[{k:r[k] for k in ('real_queries', 'preparation_seconds', 'actor_seconds',
                                                  'total_seconds', 'metrics')} for r in records]))
                stream.flush()
                os.fsync(stream.fileno())
            verify_sources(sources)
            write(output / 'updates.json', metadata)
            steps = [r['actor_update'] for r in metadata]
            summary = dict(recorded_updates=len(steps), first_update=steps[0], last_update=steps[-1],
                           contiguous=steps==list(range(steps[0],steps[-1]+1)), query_occurrences=count,
                           rank_replicas_counted_once=True, terminal_exit_code=transport['state']['ExitCode'])
            write(complete, dict(schema='recorded-training-trajectory-publication-v1', provenance=provenance,
                sources=sources, outputs={p:sha(output/p) for p in ('trajectory.jsonl','updates.json')},
                summary=summary, container_id=transport['container_id'], image=transport['image'],
                training_success_certified=False,
                scope='Recorded terminal collection only; query occurrences include padding. No gold/reward reconstruction, action resampling, full training audit or efficacy claim.'))
            write(output / 'status.json', dict(phase='complete', **summary))
            return dict(status='complete', **summary)
        except Exception as exc:
            write(output / 'status.json', dict(phase='failed_preserve_evidence', error=repr(exc), automatic_retry=False))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', default=os.environ.get('JEV_LOCAL_CONFIG', '.local/config/macros.json'))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.local)))


if __name__ == '__main__':
    main()

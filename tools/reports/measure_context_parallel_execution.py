"""Measure the shared execution layer against the serial path (CPU, read-only).

Paired single measurements, not a statistics suite: every number here is one
serial run and one parallel run over the same items, plus repeat release loads
through the content-keyed handle. Equivalence is asserted on the outputs, so a
faster result that differs is a failure, not a speed-up.

The CPU grid exists because the first measured configuration lost: one IPC round
trip per item on ~1.75 ms of work ran at 0.36x the serial path. A width is only
adopted for a workload when its own chunked measurement beats serial.
"""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import time

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_execution import map_cpu
from multimodal_jev.context_release_handle import ReleaseHandle
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config

SCHEMA = 'jev-context-parallel-execution-measurement-v1'


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def seconds(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def bind_row(item):
    """Compile one candidate request and bind its exclusive-event targets."""
    from multimodal_jev.context_candidate_task import bind_candidate_targets, compile_candidates
    row, candidates, requirements = item
    request = compile_candidates(row['audio'], candidates, requirements,
                                 semantics='exclusive_event')
    bound = bind_candidate_targets(request, semantics='exclusive_event',
                                   exclusive_event=row['intent'])
    return dict(request_sha256=bound['request_sha256'], labels=list(bound['labels']),
                supervised=list(bound['supervised']))


def release_probe(c, handle):
    root = Path(c['release'])
    digest = file_hash(root/'MANIFEST.json')
    loads = [seconds(lambda: handle.load(root, digest, c['role'], c['purpose']))
             for _ in range(3)]
    evidence = [value for value, _ in loads]
    timings = [elapsed for _, elapsed in loads]
    reused = [value[3] for value in evidence]
    if reused[0] or not all(reused[1:]):
        raise ValueError('only the first load in a fresh handle may authenticate')
    payload = canonical(dict(data=evidence[0][0], task=evidence[0][1],
                             manifest_schema=evidence[0][2]['schema']))
    for data, task, manifest, _ in evidence[1:]:
        if canonical(dict(data=data, task=task, manifest_schema=manifest['schema'])) != payload:
            raise ValueError('reused release evidence differs from the authenticated load')
    return dict(manifest_sha256=digest, schema=evidence[0][2]['schema'],
                rows=len(evidence[0][0]),
                candidates=len(evidence[0][1]['schema']['candidates']),
                authenticate_seconds=timings[0], first_reuse_seconds=timings[1],
                repeat_seconds=timings[2], outputs_sha256=payload, reused=reused,
                reauthentication_saved_fraction=1 - timings[1]/timings[0],
                stats=handle.stats())


def compact(stats):
    """Keep the pool statistics but not 512 per-task floats in the report."""
    durations = stats.pop('task_seconds')
    stats['task_seconds_count'] = len(durations)
    stats['task_seconds_sha256'] = canonical(durations)
    if durations:
        stats['task_seconds_min'] = min(durations)
        stats['task_seconds_max'] = max(durations)
    return stats


def candidate_probe(c, handle):
    root = Path(c['release'])
    digest = file_hash(root/'MANIFEST.json')
    data, task, _, _ = handle.load(root, digest, c['role'], c['purpose'])
    rows = data[:c['sample_rows']]
    if len(rows) != c['sample_rows']:
        raise ValueError('sample_rows exceeds the admitted learning population')
    candidates = [dict(id=x['id'], definition=x['description'])
                  for x in task['schema']['candidates']]
    requirements = list(task['thinking']['requirements'])
    items = [(row, candidates, requirements) for row in rows]
    serial, serial_seconds = seconds(lambda: [bind_row(item) for item in items])
    expected = canonical(serial)
    grid = []
    for entry in c['cpu_grid']:
        (parallel, stats), elapsed = seconds(
            lambda: map_cpu(bind_row, items, workers=entry['cpu_workers'],
                            window=entry['window'], chunk=entry['chunk'],
                            min_parallel=entry['min_parallel']))
        if canonical(parallel) != expected:
            raise ValueError('parallel candidate binding differs from serial')
        grid.append(dict(entry, parallel_seconds=elapsed,
                         speedup=serial_seconds/elapsed, equivalent=True,
                         stats=compact(stats)))
    return dict(sample_rows=len(rows), candidates=len(candidates),
                serial_seconds=serial_seconds, outputs_sha256=expected,
                equivalent=True, grid=grid,
                best=None if not grid else min(grid, key=lambda row: row['parallel_seconds']))


def environment():
    result = dict(python=sys.version.split()[0], host=platform.node(),
                  machine=platform.machine(), cpu_count=os.cpu_count())
    try:
        import torch
        result['torch'] = torch.__version__
    except ImportError:
        result['torch'] = None
    return result


def acceptance(report, c):
    """Deterministic verdict; every clause is a measured fact, not a judgement."""
    reuse = report.get('release_reuse')
    binding = report.get('candidate_binding')
    checks = [
        dict(check='release_authenticated_once', passed=bool(reuse) and reuse['reused'] == [False, True, True]),
        dict(check='reused_evidence_byte_identical', passed=bool(reuse)),
        dict(check='release_population_complete',
             passed=bool(reuse) and reuse['rows'] > 0 and reuse['candidates'] == 60),
        dict(check='every_grid_entry_equivalent',
             passed=bool(binding) and all(row['equivalent'] for row in binding['grid'])),
        dict(check='grid_covers_every_configured_width',
             passed=bool(binding) and len(binding['grid']) == len(c['cpu_grid'])),
        dict(check='model_calls_zero', passed=report['model_calls'] == 0),
        dict(check='npu_not_executed', passed=report['npu_executed'] is False),
    ]
    best = None if not binding else binding['best']
    return dict(verdict='accepted' if all(x['passed'] for x in checks) else 'rejected',
                checks=checks, adopted=best,
                note='adoption is per workload: a width is used only where its own '
                     'chunked measurement beats serial on the same items')


def run(c, provenance):
    if c['schema'] != SCHEMA:
        raise ValueError('explicit execution measurement schema required')
    if not 1 <= c['sample_rows']:
        raise ValueError('positive sample required')
    if len({(e['cpu_workers'], e['window'], e['chunk'], e['min_parallel'])
            for e in c['cpu_grid']}) != len(c['cpu_grid']):
        raise ValueError('duplicate cpu grid entry')
    out = Path(c['output'])
    out.mkdir(parents=True, exist_ok=False)
    commit_json(out/'config.json', c)
    commit_json(out/'provenance.json', provenance)
    handle = ReleaseHandle()
    report = dict(schema=SCHEMA, status='measured', model_calls=0, npu_executed=False,
                  environment=environment(), implementation_sha256=file_hash(__file__),
                  execution_sha256=file_hash(Path(map_cpu.__code__.co_filename)),
                  boundary='paired single measurement; not a repeated-trial benchmark')
    for probe in c['probes']:
        started = time.perf_counter()
        if probe == 'release_reuse':
            report[probe] = release_probe(c, handle)
        elif probe == 'candidate_binding':
            report[probe] = candidate_probe(c, handle)
        else:
            raise ValueError('unregistered probe: ' + probe)
        report[probe]['wall_seconds'] = time.perf_counter() - started
        commit_json(out/f'{probe}.json', report[probe])
    report['acceptance'] = acceptance(report, c)
    report['status'] = report['acceptance']['verdict']
    commit_json(out/'report.json', report)
    return report


def summary(report):
    lines = [f"status={report['status']} model_calls={report['model_calls']} "
             f"npu_executed={report['npu_executed']}"]
    reuse = report.get('release_reuse')
    if reuse:
        lines.append(f"release_reuse rows={reuse['rows']} authenticate={reuse['authenticate_seconds']:.3f}s "
                     f"reuse={reuse['first_reuse_seconds']:.3f}s saved={reuse['reauthentication_saved_fraction']:.1%} "
                     f"reused={reuse['reused']} outputs={reuse['outputs_sha256'][:16]}")
    binding = report.get('candidate_binding')
    if binding:
        lines.append(f"candidate_binding rows={binding['sample_rows']} "
                     f"serial={binding['serial_seconds']:.3f}s outputs={binding['outputs_sha256'][:16]}")
        for row in binding['grid']:
            lines.append(f"  workers={row['cpu_workers']} window={row['window']} chunk={row['chunk']} "
                         f"min_parallel={row['min_parallel']} parallel={row['parallel_seconds']:.3f}s "
                         f"speedup={row['speedup']:.2f}x equivalent={row['equivalent']}")
    verdict = report.get('acceptance')
    if verdict:
        failed = [x['check'] for x in verdict['checks'] if not x['passed']]
        lines.append(f"verdict={verdict['verdict']} failed={failed or 'none'}")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    args = parser.parse_args()
    c, provenance = load_config(args.config, args.local)
    print(summary(run(c, provenance)))


if __name__ == '__main__':
    main()

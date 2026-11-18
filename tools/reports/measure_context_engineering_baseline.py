"""Deterministic CPU baseline for the engineering-base componentization.

Measures the serial cost of the shared data path before any parallel layer is
introduced: full release re-authentication, hashing throughput at several I/O
widths, candidate-request compilation and target binding. Read-only, no model
calls, no NPU. Numbers are a single paired measurement, not a benchmark suite,
and must be re-measured on the same workload after each candidate.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config

SCHEMA = 'context-engineering-baseline-v1'


def seconds(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def release_load(c):
    from multimodal_jev.slurp_portable_release import load
    root = Path(c['release'])
    digest = file_hash(root/'MANIFEST.json')
    (data, task, manifest), elapsed = seconds(
        lambda: load(root, digest, c['role'], purpose=c['purpose']))
    return dict(seconds=elapsed, manifest_sha256=digest, schema=manifest['schema'],
                rows=len(data), candidates=len(task['schema']['candidates'])), data, task


def release_inventory(c):
    root = Path(c['inventory_root'])
    files = sorted(p for p in root.rglob('*') if p.is_file())
    if len(files) < c['inventory_max_files']:
        raise ValueError('inventory sample exceeds available files')
    files = files[:c['inventory_max_files']]
    repeats = c['inventory_repeats']
    if type(repeats) is not int or not 1 <= repeats <= 16:
        raise ValueError('bounded inventory repeats required')
    total = sum(p.stat().st_size for p in files)
    rows = []
    for width in c['io_workers']:
        # Same bounded file set at every width; every byte re-read on every pass.
        passes = []
        for _ in range(repeats):
            _, elapsed = seconds(lambda: _hash_files(files, width))
            passes.append(elapsed)
        best = min(passes)
        rows.append(dict(io_workers=width, files=len(files), bytes=total,
                         passes_seconds=passes, best_seconds=best,
                         mib_per_second=total / best / 1024**2))
    return dict(root=str(root), files=len(files), bytes=total, repeats=repeats, rows=rows)


def _hash_files(files, width):
    from concurrent.futures import ThreadPoolExecutor
    from multimodal_jev.parallel_evidence import workers
    workers(width)

    def digest(path):
        return file_hash(path)

    # Read every byte on every pass; ordered results are irrelevant here.
    with ThreadPoolExecutor(max_workers=width) as pool:
        return list(pool.map(digest, files))


def candidate_path(c, data, task):
    from multimodal_jev import context_candidate_task as module
    from multimodal_jev.context_candidate_task import bind_candidate_targets, compile_candidates
    candidates = [dict(id=x['id'], definition=x['description']) for x in task['schema']['candidates']]
    requirements = list(task['thinking']['requirements'])
    rows = data[:c['sample_rows']]
    if len(rows) != c['sample_rows']:
        raise ValueError('sample_rows exceeds the admitted learning population')
    compiled, compile_seconds = seconds(lambda: [
        compile_candidates(row['audio'], candidates, requirements, semantics='exclusive_event')
        for row in rows])
    bound, bind_seconds = seconds(lambda: [
        bind_candidate_targets(request, semantics='exclusive_event', exclusive_event=row['intent'])
        for request, row in zip(compiled, rows, strict=True)])
    return dict(sample_rows=len(rows), candidates=len(candidates),
                compile_seconds=compile_seconds, bind_seconds=bind_seconds,
                per_row_seconds=(compile_seconds + bind_seconds) / len(rows),
                supervised_rows=sum(all(b['supervised']) for b in bound),
                implementation_sha256=file_hash(module.__file__))


def environment():
    result = dict(python=sys.version.split()[0], host=platform.node(),
                  machine=platform.machine(), cpu_count=os.cpu_count())
    try:
        import torch
        result['torch'] = torch.__version__
    except ImportError:
        result['torch'] = None
    return result


def run(c, provenance):
    if c['schema'] != SCHEMA:
        raise ValueError('explicit baseline schema required')
    # Shared release/inventory roots are read-only; only the report root must be
    # admitted local storage (commit_json re-checks every written path).
    out = Path(c['output'])
    out.mkdir(parents=True, exist_ok=False)
    commit_json(out/'config.json', c)
    commit_json(out/'provenance.json', provenance)
    report = dict(schema=SCHEMA, status='measured', model_calls=0, npu_executed=False,
                  environment=environment(), implementation_sha256=file_hash(__file__))
    loaded = None
    for probe in c['probes']:
        started = time.perf_counter()
        if probe == 'release_load':
            report['release_load'], data, task = release_load(c)
            loaded = (data, task)
        elif probe == 'release_inventory':
            report['release_inventory'] = release_inventory(c)
        elif probe == 'candidate_path':
            if loaded is None:
                raise ValueError('candidate_path requires release_load earlier in probes')
            report['candidate_path'] = candidate_path(c, *loaded)
        else:
            raise ValueError('unregistered probe: ' + probe)
        report[probe]['wall_seconds'] = time.perf_counter() - started
        commit_json(out/f'{probe}.json', report[probe])
    commit_json(out/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    args = parser.parse_args()
    c, provenance = load_config(args.config, args.local)
    print(json.dumps(run(c, provenance), indent=2))


if __name__ == '__main__':
    main()

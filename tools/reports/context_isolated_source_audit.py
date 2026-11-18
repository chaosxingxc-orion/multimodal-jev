"""Owned persistent CPU process for unchanged source/selection audits.

The scheduler waits on a Future without competing for the audit worker's GIL.
No NPU dispatch, global monkey patch, automatic retry, or filesystem PASS cache.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.parallel_evidence import workers
from tools.reports.context_parallel_batch_audit import read_bound
from tools.reports.context_shared_source_audit import SourceAuditEngine

_ENGINE = None


def initialize(cpu_workers, io_workers):
    global _ENGINE
    import torch
    torch.set_num_threads(1)
    _ENGINE = SourceAuditEngine(cpu_workers=cpu_workers, io_workers=io_workers)


def execute(action, args):
    if _ENGINE is None:
        raise RuntimeError('owned audit worker not initialized')
    if action == 'statistics':
        return dict(pid=os.getpid(), batch_computations=_ENGINE.store.executions,
                    batch_reuses=_ENGINE.store.hits, events=_ENGINE.events)
    if action in ('build', 'reconstruct', 'audit'):
        return getattr(_ENGINE, action)(*args)
    if action in ('prepare_selection', 'audit_selection'):
        method = 'prepare' if action == 'prepare_selection' else 'audit'
        return getattr(_ENGINE.preparation(), method)(*args)
    raise ValueError('unregistered CPU audit action')


class IsolatedSourceAuditEngine:
    """One process per explicit controller; serialize bounded submissions.

    Nested original batch pool uses cpu_workers processes, each torch thread=1.
    Any worker exception poisons this engine. No implicit restart or retry.
    """
    def __init__(self, *, cpu_workers=4, io_workers=4):
        workers(cpu_workers); workers(io_workers)
        self.pool = ProcessPoolExecutor(max_workers=1,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize, initargs=(cpu_workers, io_workers))
        self.lock = threading.Lock()
        self.failed = False
        self.closed = False

    def call(self, action, *args):
        with self.lock:
            if self.closed or self.failed:
                raise RuntimeError('audit worker closed or failed; automatic retry forbidden')
            try:
                return self.pool.submit(execute, action, args).result()
            except BaseException:
                self.failed = True
                raise

    def build(self, c, provenance):
        return self.call('build', c, provenance)

    def reconstruct(self, c):
        return self.call('reconstruct', c)

    def audit(self, c):
        return self.call('audit', c)

    def preparation(self):
        return SimpleNamespace(
            prepare=lambda c,p:self.call('prepare_selection', c, p),
            audit=lambda c,p:self.call('audit_selection', c, p))

    def statistics(self):
        return self.call('statistics')

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                self.pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def run(c, provenance):
    root = Path(c['output'])
    root.mkdir(parents=True, exist_ok=False)
    commit_json(root/'config.json', c)
    commit_json(root/'provenance.json', provenance)
    try:
        for path, digest in c['pins'].items():
            if file_hash(path) != digest:
                raise ValueError('isolated audit code changed: '+path)
        expected = read_bound(c['source_plan'])
        finalization = read_bound(c['source_finalization'])
        selection_spec = read_bound(c['selection_spec'])
        selection_expected = read_bound(c['selection_expected'])
        times = []
        with IsolatedSourceAuditEngine(cpu_workers=c['cpu_workers'], io_workers=c['io_workers']) as engine:
            for i in range(2):
                start = time.perf_counter()
                actual = engine.build(finalization, expected['config_provenance'])
                times.append(time.perf_counter()-start)
                if actual != expected:
                    raise ValueError('isolated full source proof differs')
                commit_json(root/f'source-plan-{i}.json', actual)
            start = time.perf_counter()
            actual = engine.preparation().audit(selection_spec, selection_expected['provenance'])
            selection_seconds = time.perf_counter()-start
            if actual != selection_expected:
                raise ValueError('isolated actual selection package proof differs')
            commit_json(root/'selection-audit.json', actual)
            stats = engine.statistics()
            if stats['pid'] == os.getpid() or stats['batch_computations'] != 1 or stats['batch_reuses'] < 3:
                raise ValueError('isolated persistent process and shared full batch proof required')
        # Closing the pool waits for owned process termination before publication.
        report = dict(status='isolated_source_selection_audited', full_source_equal=True,
            full_selection_equal=True, source_seconds=times, selection_seconds=selection_seconds,
            worker=stats, worker_closed=True, parent_pid=os.getpid(), model_calls=0,
            npu_executed=False, training_dispatch_allowed=False)
        commit_json(root/'report.json', report)
        commit_json(root/'terminal.json', dict(exit_code=0, status=report['status']))
        return report
    except BaseException as exc:
        commit_json(root/'terminal.json', dict(exit_code=1, error=repr(exc), restart=False))
        raise


if __name__ == '__main__':
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--local', required=True)
    args = p.parse_args()
    c, provenance = load_config(args.config, args.local)
    print(json.dumps(run(c, provenance), indent=2))

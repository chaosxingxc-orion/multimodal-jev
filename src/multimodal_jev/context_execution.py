"""One bounded parallel layer for preprocessing and audit work.

Two kinds of work, two pools, never mixed:

* ``map_io`` — hashing, reading, JSON parsing. Bounded thread pool; Python
  releases the GIL for these, so threads help and process overhead does not.
* ``map_cpu`` — Python-dense loops (target compilation, batch recomputation).
  Spawn process pool, one thread per child, contiguous items dispatched in
  configurable chunks so a worker does not pay one IPC round trip per item.

Both preserve input order and stop at the first failure, cancelling work that
has not started. Neither samples: the caller still passes the complete item
list. Small panels stay serial because process start-up costs more than the
work — that floor is a configured number, not a hidden heuristic.

Content-addressed reuse is not reimplemented here: ``singleflight`` returns the
existing ``AuditSingleFlight`` so every caller shares one eviction policy.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
import multiprocessing
import os
import sys
import threading
import time

from .audit_singleflight import AuditSingleFlight
from .parallel_evidence import workers as _workers

_CPU_FUNCTION = None


def width(value):
    """Validate one concurrency setting; same range as file evidence."""
    return _workers(value)


def widths(config):
    """Read the two widths from config instead of hardcoding 4 and 2."""
    if type(config) is not dict:
        raise ValueError('object execution config required')
    if set(config) != {'io_workers', 'cpu_workers'}:
        raise ValueError('explicit io_workers and cpu_workers required')
    return dict(io_workers=width(config['io_workers']), cpu_workers=width(config['cpu_workers']))


class Stats:
    """Wall clock, per-dispatch seconds and peak in-flight tasks for one pool run.

    ``task_seconds`` holds one entry per *dispatch unit*: an item when work goes
    out one at a time, a whole chunk when it is batched. ``total`` always counts
    items, and ``unit`` names what a duration refers to, so a chunked run cannot
    be misread as per-item timings.
    """

    def __init__(self, kind, width, total, parallel, unit='item'):
        if kind not in ('io', 'cpu') or type(total) is not int or total < 0:
            raise ValueError('explicit pool kind and nonnegative task count required')
        if unit not in ('item', 'chunk'):
            raise ValueError('explicit dispatch unit required')
        self.kind, self.width, self.total, self.parallel = kind, width, total, parallel
        self.unit = unit
        self.lock = threading.Lock()
        self.durations = []
        self.peak_in_flight = 0
        self.started = time.perf_counter()

    def observe_in_flight(self, count):
        with self.lock:
            self.peak_in_flight = max(self.peak_in_flight, count)

    def task(self, seconds):
        if type(seconds) not in (int, float) or seconds < 0:
            raise ValueError('nonnegative task duration required')
        with self.lock:
            self.durations.append(float(seconds))

    def to_dict(self):
        elapsed = time.perf_counter() - self.started
        return dict(kind=self.kind, width=self.width, total=self.total,
                    parallel=self.parallel, unit=self.unit, wall_seconds=elapsed,
                    task_seconds=list(self.durations),
                    task_seconds_total=sum(self.durations),
                    peak_in_flight=self.peak_in_flight,
                    speedup=sum(self.durations) / elapsed if elapsed > 0 else None)


def _bounded(executor, fn, items, window, stats):
    """Submit at most ``window`` tasks ahead; collect results in input order."""
    results = [None] * len(items)
    pending = deque()
    source = iter(range(len(items)))
    try:
        for index in source:
            pending.append((index, executor.submit(fn, items[index])))
            stats.observe_in_flight(len(pending))
            if len(pending) >= window:
                break
        while pending:
            index, future = pending.popleft()
            results[index] = future.result()
            for following in source:
                pending.append((following, executor.submit(fn, items[following])))
                stats.observe_in_flight(len(pending))
                break
    except BaseException:
        # Stop starting new work; callbacks already running finish on shutdown.
        for _, future in pending:
            future.cancel()
        raise
    return results


def _timed(fn, stats):
    def call(item):
        start = time.perf_counter()
        try:
            return fn(item)
        finally:
            stats.task(time.perf_counter() - start)
    return call


def map_io(fn, items, *, workers, ahead=None):
    """Ordered bounded thread map for I/O-bound work; returns (results, stats)."""
    width_ = width(workers)
    items = list(items)
    window = width_ if ahead is None else max(width_, width(ahead))
    parallel = width_ > 1 and len(items) > 1
    stats = Stats('io', width_, len(items), parallel)
    if not parallel:
        results = []
        for item in items:
            start = time.perf_counter()
            results.append(fn(item))
            stats.task(time.perf_counter() - start)
        return results, stats.to_dict()
    with ThreadPoolExecutor(max_workers=width_, thread_name_prefix='jev-io') as pool:
        results = _bounded(pool, _timed(fn, stats), items, window, stats)
    return results, stats.to_dict()


def _cpu_init(fn, initializer, initargs):
    global _CPU_FUNCTION
    # One thread per child, set by environment only: importing torch here would
    # charge every pool start with a multi-second import the payload may not use.
    os.environ['OMP_NUM_THREADS'] = '1'
    if initializer is not None:
        initializer(*initargs)
    torch = sys.modules.get('torch')
    if torch is not None:
        torch.set_num_threads(1)
    _CPU_FUNCTION = fn


def _cpu_chunk(block):
    start = time.perf_counter()
    values = [_CPU_FUNCTION(item) for item in block]
    return values, time.perf_counter() - start


def map_cpu(fn, items, *, workers, initializer=None, initargs=(), window=None,
            min_parallel=16, chunk=1):
    """Ordered bounded process map for Python-dense work; returns (results, stats).

    ``fn`` and ``initializer`` must be picklable module-level callables because
    the pool uses the spawn context. Contiguous items are dispatched ``chunk`` at
    a time so per-item IPC does not dominate: at 1.75 ms per item, one round trip
    per item measured 0.36x the serial path, while the same work chunked beats it.

    Panels smaller than ``min_parallel`` run serially in this process: the
    measured 8-batch panel was slower in parallel, so the floor sits above it, and
    the real crossover is measured per workload.
    """
    width_ = width(workers)
    if type(min_parallel) is not int or min_parallel < 1:
        raise ValueError('positive integer parallel threshold required')
    if type(chunk) is not int or chunk < 1:
        raise ValueError('positive integer chunk size required')
    items = list(items)
    if window is None:
        window = width_
    else:
        window = width(window)
    parallel = width_ > 1 and len(items) >= min_parallel
    stats = Stats('cpu', width_, len(items), parallel, unit='chunk' if parallel else 'item')
    if not parallel:
        results = []
        for item in items:
            start = time.perf_counter()
            results.append(fn(item))
            stats.task(time.perf_counter() - start)
        return results, stats.to_dict()
    blocks = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    with ProcessPoolExecutor(max_workers=width_, mp_context=multiprocessing.get_context('spawn'),
                             initializer=_cpu_init,
                             initargs=(fn, initializer, tuple(initargs))) as pool:
        wrapped = _bounded(pool, _cpu_chunk, blocks, window, stats)
    results = []
    for values, seconds in wrapped:
        results.extend(values)
        stats.task(seconds)
    return results, stats.to_dict()


def singleflight(*, max_entries=4, max_running=1):
    """Shared content-addressed audit reuse; never a reimplementation."""
    return AuditSingleFlight(max_entries=max_entries, max_running=max_running)


def content_key(*parts):
    """Digest complete inputs (config, manifests, code identity) into a cache key."""
    if not parts:
        raise ValueError('at least one input part required')
    payload = json.dumps(parts, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()
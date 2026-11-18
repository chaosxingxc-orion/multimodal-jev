"""Aggregate complete rank timings without confusing update time with throughput."""
import math
from .simple_batch_config import batch_layout


def summarize(config, reports):
    layout = batch_layout(config)
    if (len(reports) != 4 or {r['rank'] for r in reports} != set(range(4))
            or any(r.get('passed') is not True for r in reports)):
        raise ValueError('all four successful rank reports required')
    start = config.get('start_update', 0)
    expected = list(range(start+1, start+config['updates']+1))
    if any([s['update'] for s in r['step_metrics']] != expected for r in reports):
        raise ValueError('complete aligned update ledgers required')
    if len({r['final_adapter_id'] for r in reports}) != 1:
        raise ValueError('ranks disagree on final exported policy')
    steps = []
    for index, update in enumerate(expected):
        records = [r['step_metrics'][index] for r in reports]
        counts = {r['admission']['global_real_requests'] for r in records}
        if (len(counts) != 1 or type(next(iter(counts))) is not int
                or not 0 < next(iter(counts)) <= layout['global_batch']):
            raise ValueError('common positive real request count required')
        for r in records:
            if (r['batch_layout'] != layout
                    or r['admission']['global_rows'] != layout['global_batch']
                    or r['admission']['reduction'] != 'real_request_mean'):
                raise ValueError('exact runtime batch and real-count reduction required')
            for name in ('seconds','preparation_seconds','audit_seconds','total_step_seconds'):
                v = r[name]
                if type(v) not in (int,float) or not math.isfinite(v) or v < 0:
                    raise ValueError('finite nonnegative stage timings required')
            checkpoint = r.get('checkpoint_seconds', 0.)
            if (type(checkpoint) not in (int,float) or not math.isfinite(checkpoint) or checkpoint < 0
                    or r['total_step_seconds'] <= 0
                    or r['total_step_seconds']+1e-6 < r['seconds']+r['preparation_seconds']+r['audit_seconds']+checkpoint):
                raise ValueError('total timing must contain every measured phase')
            if (type(r['peak_memory_allocated']) is not int or type(r['peak_memory_reserved']) is not int
                    or not 0 < r['peak_memory_allocated'] <= r['peak_memory_reserved']):
                raise ValueError('valid allocated and reserved memory peaks required')
            if r.get('frozen_audit') != config.get('frozen_audit','every_update'):
                raise ValueError('frozen audit policy differs')
            if ((config.get('frozen_audit','every_update') == 'every_update' or 'adapter_id' in r)
                    and r.get('frozen_hash_checked') is not True):
                raise ValueError('required frozen base audit absent')
        # Ranks execute concurrently; summing rank durations would undercount
        # throughput. Max phase times are diagnostics, not an additive wallclock.
        real = next(iter(counts))
        seconds = max(r['total_step_seconds'] for r in records)
        steps.append(dict(update=update,real_requests=real,seconds=seconds,
                          real_requests_per_second=real/seconds,
                          maximum_rank_update_seconds=max(r['seconds'] for r in records),
                          maximum_rank_preparation_seconds=max(r['preparation_seconds'] for r in records),
                          maximum_rank_audit_seconds=max(r['audit_seconds'] for r in records),
                          maximum_rank_checkpoint_seconds=max(r.get('checkpoint_seconds',0.) for r in records)))
    real = sum(s['real_requests'] for s in steps)
    seconds = sum(s['seconds'] for s in steps)
    return dict(passed=True,batch_layout=layout,updates=len(steps),real_requests=real,
                measured_step_seconds=seconds,real_requests_per_second=real/seconds,steps=steps,
                maximum_allocated_bytes=max(s['peak_memory_allocated'] for r in reports for s in r['step_metrics']),
                maximum_reserved_bytes=max(s['peak_memory_reserved'] for r in reports for s in r['step_metrics']),
                scope='Complete rank timing aggregation; excludes startup, tests and external evaluation. Not independent checkpoint/parity acceptance or full-dataset stability.')

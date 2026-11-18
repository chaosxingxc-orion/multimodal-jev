"""Aggregate actual per-rank verl contributions at a synchronized update.

The registered objective already scales each local numerator by DP size and
the global real-request denominator. Sum micro contributions, then average
ranks; do not average the micro means a second time.
"""
import math


def aggregate_update(records):
    if len(records) != 4 or len({r['update'] for r in records}) != 1:
        raise ValueError('four rank records for one update required')
    layouts = [r['batch_layout'] for r in records]
    if any(layout != layouts[0] for layout in layouts):
        raise ValueError('rank batch layouts differ')
    layout = layouts[0]
    if layout['world_size'] != 4 or layout['global_batch'] != 128:
        raise ValueError('DP4 global128 metrics required')
    admissions = [r['admission'] for r in records]
    if any(a != admissions[0] for a in admissions):
        raise ValueError('rank population normalization differs')
    admission = admissions[0]
    real = admission['global_real_requests']
    if admission['stage'] != 'instruction' or type(real) is not int or not 0 < real <= 128:
        raise ValueError('explicit instruction real-request population required')
    contributions = []
    for record in records:
        values = record['metrics']['jev/ce_contribution']
        if (len(values) != layout['accumulation_per_rank']
                or any(not math.isfinite(v) or v < 0 for v in values)):
            raise ValueError('complete finite CE micro contributions required')
        if any(v != 0 for v in record['metrics']['jev/rl_contribution']):
            raise ValueError('instruction metrics cannot contain RL contributions')
        contributions.append(sum(values))
        for key in ('seconds', 'total_step_seconds'):
            if not math.isfinite(record[key]) or record[key] <= 0:
                raise ValueError('finite positive step timing required')
    slowest = max(r['seconds'] for r in records)
    end_to_end = max(r['total_step_seconds'] for r in records)
    return dict(update=records[0]['update'], real_requests=real,
                weighted_ce_real_request_mean=sum(contributions) / 4,
                slowest_update_seconds=slowest, slowest_total_step_seconds=end_to_end,
                update_requests_per_second=real / slowest,
                total_step_requests_per_second=real / end_to_end,
                peak_allocated_bytes=max(r['peak_memory_allocated'] for r in records),
                peak_reserved_bytes=max(r['peak_memory_reserved'] for r in records),
                scope='Observed training batch CE and timing; no heldout accuracy or steady-state claim.')

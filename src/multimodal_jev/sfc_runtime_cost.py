"""Input-stratified, descriptive cost projection; never a dispatch budget."""
from collections import Counter,defaultdict
import math


def finite_seconds(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:
        raise ValueError('finite nonnegative measured seconds required')
    return value


def stratum(row,decision):
    role=decision['id'].rsplit('.',1)[-1]
    if role not in ('present','absent','calls'):
        raise ValueError('known SFC Jev decision role required')
    return f"audio_turns={len(row['audio_sha256'])}/{role}"


def project(observed_rows,predictions,elapsed_seconds,rows,plan):
    """Project all decisions, stratifying by supplied audio turns and answer kind.

    Only predictions from an independently accepted TRAIN run may be supplied by
    the caller. No correctness, targets or generated text are used here.
    """
    expected={(r['id'],d['id']):stratum(r,d) for r in observed_rows for d in r['input']['decisions']}
    if len(expected)!=sum(len(r['input']['decisions']) for r in observed_rows):
        raise ValueError('unique observed decisions required')
    seen=set();samples=defaultdict(list)
    for p in predictions:
        key=(p['example_id'],p['decision_id'])
        if key not in expected or key in seen:raise ValueError('exact observed decision coverage required')
        seen.add(key);samples[expected[key]].append(finite_seconds(p['seconds']))
    if seen!=set(expected) or not seen:raise ValueError('complete nonempty timing sample required')
    elapsed=finite_seconds(elapsed_seconds);generation=math.fsum(t for values in samples.values() for t in values)
    if elapsed<generation:raise ValueError('elapsed runtime cannot be shorter than measured generation')
    overhead=elapsed-generation
    by_id={r['id']:r for r in rows}
    registered=[uid for shard in plan['shards'] for uid in shard['ordered_bundle_ids']]
    if len(by_id)!=len(rows) or len(set(registered))!=len(registered) or set(registered)!=set(by_id):
        raise ValueError('exact complete planned bundle coverage required')
    stats={k:dict(observations=len(v),mean_seconds=math.fsum(v)/len(v),
                  min_seconds=min(v),max_seconds=max(v)) for k,v in sorted(samples.items())}
    reports=[];counts=Counter()
    for shard in plan['shards']:
        subset=[by_id[uid] for uid in shard['ordered_bundle_ids']]
        cells=Counter(stratum(r,d) for r in subset for d in r['input']['decisions'])
        if sum(cells.values())!=shard['decisions'] or len(subset)!=shard['bundles']:
            raise ValueError('registered shard counts differ')
        if cells.keys()-stats.keys():raise ValueError('unobserved timing stratum; no invented fallback')
        estimates={name:math.fsum(n*stats[k][name+'_seconds'] for k,n in cells.items())+overhead
                   for name in ('mean','min','max')}
        reports.append(dict(shard_id=shard['id'],decisions=sum(cells.values()),strata=dict(cells),
            estimated_serial_seconds=estimates['mean'],observed_min_sensitivity_seconds=estimates['min'],
            observed_max_sensitivity_seconds=estimates['max'],
            mean_exceeds_86400_second_container_limit=estimates['mean']>86400,
            max_sensitivity_exceeds_86400_second_container_limit=estimates['max']>86400))
        counts.update(cells)
    if len(rows)!=plan['bundles'] or sum(counts.values())!=plan['decisions']:
        raise ValueError('complete population totals differ')
    total=math.fsum(r['estimated_serial_seconds'] for r in reports)
    return dict(observed_bundles=len(observed_rows),observed_source_groups=len({r['source_group'] for r in observed_rows}),
        observed_decisions=len(predictions),observed_elapsed_seconds=elapsed,observed_generation_seconds=generation,
        observed_residual_seconds_per_container=overhead,stratum_measurements=stats,
        population_bundles=len(rows),population_decisions=sum(counts.values()),population_strata=dict(counts),
        shard_count=len(reports),shards=reports,estimated_serial_wall_days=total/86400,
        estimated_npu_device_hours=total*2/3600,
        observed_min_sensitivity_wall_days=math.fsum(r['observed_min_sensitivity_seconds'] for r in reports)/86400,
        observed_max_sensitivity_wall_days=math.fsum(r['observed_max_sensitivity_seconds'] for r in reports)/86400,
        estimate_is_not_budget_or_completion=True,dispatched=False,
        limitations='Parent TRAIN probe only. Strata distinguish audio-turn count and present/absent/calls, '
        'not audio duration, prompt length, output length, function complexity or host load. '
        'Observed min/max are sensitivity scenarios, not confidence intervals or runtime bounds. '
        'Residual includes model loading/hashing plus per-example audio/processor work; treating it as '
        'one constant per shard may underestimate larger shards. Excludes Docker startup, queue delays, '
        'independent verification and scoring. No candidate cost or task-quality claim; no scope reduction.')

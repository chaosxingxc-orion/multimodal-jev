"""Small, deterministic admission check; never executes an evaluation.

Receipts are produced by finite performance/parity runs, outside TEST/transfer.
Hashes bind their bytes; they are evidence references, not a substitute for an
independent timing/output audit. Frozen historical dispatchers stay unchanged.
"""
import hashlib
import json
import math
from pathlib import Path


CASES = {'cold', 'warm', 'changed_audio', 'changed_definition',
         'changed_policy', 'batch_reorder', 'eviction'}


def read_receipt(binding):
    raw = Path(binding['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding['sha256']:
        raise ValueError('performance evidence hash changed')
    return json.loads(raw)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def assess(baseline, candidate, parity, runtime_sha256):
    """Compare whole campaigns; initialization is charged once per campaign.

    Reuse across jobs is allowed. Every job still consumes its full population
    under its declared policy. Neither repeated predictions nor per-job cold
    restarts are required; the wall clock includes the final CPU audit drain.
    """
    for row in (baseline, candidate):
        if (row.get('scope') != 'complete_evaluation_campaign'
                or row.get('split') not in ('train', 'selection')
                or row.get('timing_boundary') != 'before_prepare_to_collected_and_scored'
                or row.get('cache_construction_included') is not True
                or row.get('independently_audited') is not True
                or row.get('exit_code') != 0
                or type(row.get('recordings')) is not int or row['recordings'] <= 0
                or type(row.get('unique_audio_count')) is not int
                or not 0 < row['unique_audio_count'] <= row['recordings']
                or row.get('completed_recordings') != row['recordings']
                or type(row.get('device_count')) is not int or row['device_count'] != 4
                or not positive(row.get('end_to_end_seconds'))):
            raise ValueError('complete four-device campaign evidence required')
        jobs=row.get('jobs', [])
        if (not isinstance(jobs,list) or not jobs
                or any(set(j)!={'job_id','population_sha256','policy_id','prompt_layout',
                                'output_contract_sha256','recordings'} for j in jobs)
                or any(type(j['recordings']) is not int or j['recordings']<=0 for j in jobs)
                or len({j['job_id'] for j in jobs})!=len(jobs)
                or sum(j['recordings'] for j in jobs)!=row['recordings']):
            raise ValueError('complete distinct campaign jobs required')
        for stage in ('input_load','engine_start','policy_switch','inference','output_write','collection','cpu_audit'):
            value=row.get('stage_work_seconds',{}).get(stage)
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                raise ValueError('campaign stage timing required: '+stage)
    for key in ('jobs', 'unique_audio_count', 'recordings', 'device_count', 'split', 'timing_boundary'):
        if not baseline.get(key) or baseline[key] != candidate.get(key):
            raise ValueError('matched performance workload required: ' + key)
    if not runtime_sha256 or candidate.get('runtime_sha256') != runtime_sha256:
        raise ValueError('performance evidence belongs to another runtime')
    for key in ('logit_max_abs', 'probability_max_abs', 'threshold_flips', 'top1_changes'):
        if type(parity.get(key)) not in (int, float) or parity[key] != 0:
            raise ValueError('numeric zero parity errors required')
    if not baseline.get('runtime_sha256'):
        raise ValueError('frozen baseline runtime identity required')
    if (parity.get('runtime_sha256') != runtime_sha256
            or parity.get('baseline_runtime_sha256') != baseline.get('runtime_sha256')
            or set(parity.get('policy_ids',[])) != {j['policy_id'] for j in candidate['jobs']}
            or parity.get('independently_audited') is not True
            or parity.get('existing_thresholds_preserved') is not True
            or parity.get('logit_max_abs') != 0 or parity.get('probability_max_abs') != 0
            or parity.get('threshold_flips') != 0 or parity.get('top1_changes') != 0
            or set(parity.get('cases_passed', [])) != CASES
            or parity.get('native_kv_reads_and_writes') is not True
            or parity.get('prefix_cache_hits_verified') is not True
            or parity.get('history_restoration_verified') is not True):
        raise ValueError('frozen-runtime parity and cache isolation evidence required')
    speedup = baseline['end_to_end_seconds'] / candidate['end_to_end_seconds']
    if speedup < 10:
        raise ValueError(f'end-to-end speedup {speedup:.6g} is below 10x')
    return dict(status='performance_gate_passed', measured_speedup=speedup,
                runtime_sha256=runtime_sha256, benchmark_split=candidate['split'],
                benchmark_recordings=candidate['recordings'],
                target_test_or_transfer_score_available=False)


def require_performance_admission(config):
    spec = config.get('performance_admission')
    if not isinstance(spec, dict):
        raise ValueError('10x performance admission required before formal evaluation dispatch')
    proof = read_receipt(spec['evidence'])
    return assess(*(read_receipt(proof[key]) for key in ('baseline', 'candidate', 'parity')),
                  runtime_sha256=spec['runtime_sha256'])

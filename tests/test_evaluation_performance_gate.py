import hashlib
import json

import pytest

from multimodal_jev.evaluation_performance_gate import (
    CASES, assess, require_performance_admission,
)


def evidence():
    baseline = dict(scope='complete_evaluation_campaign', split='selection',
                    timing_boundary='before_prepare_to_collected_and_scored',
                    cache_construction_included=True, independently_audited=True,
                    exit_code=0, recordings=9226, unique_audio_count=9226,
                    completed_recordings=9226, device_count=4, end_to_end_seconds=1000,
                    runtime_sha256='old',
                    jobs=[dict(job_id='point-1',population_sha256='population',policy_id='policy',
                               prompt_layout='frozen',output_contract_sha256='outputs',recordings=9226)],
                    stage_work_seconds={s:1 for s in ('input_load','engine_start','policy_switch',
                        'inference','output_write','collection','cpu_audit')})
    candidate = dict(baseline, end_to_end_seconds=100, runtime_sha256='new')
    parity = dict(runtime_sha256='new', baseline_runtime_sha256='old', policy_ids=['policy'],
                  independently_audited=True, existing_thresholds_preserved=True,
                  logit_max_abs=0, probability_max_abs=0, threshold_flips=0, top1_changes=0,
                  cases_passed=sorted(CASES), native_kv_reads_and_writes=True,
                  prefix_cache_hits_verified=True, history_restoration_verified=True)
    return baseline, candidate, parity


def test_tenfold_boundary():
    assert assess(*evidence(), 'new')['measured_speedup'] == 10


@pytest.mark.parametrize('field,value', [
    ('end_to_end_seconds', 100.001), ('end_to_end_seconds', float('nan')),
    ('scope', 'repeated_warm_requests'), ('cache_construction_included', False),
    ('unique_audio_count', 2), ('completed_recordings', 9000),
    ('device_count', 8), ('jobs', []), ('stage_work_seconds', {}),
    ('runtime_sha256', 'unmeasured'), ('split', 'test'),
    ('timing_boundary', 'container_only'), ('independently_audited', False),
])
def test_reject_unfair_comparison(field, value):
    baseline, candidate, parity = evidence()
    candidate[field] = value
    with pytest.raises(ValueError):
        assess(baseline, candidate, parity, 'new')


@pytest.mark.parametrize('field,value', [
    ('logit_max_abs', .71875), ('probability_max_abs', .0115),
    ('cases_passed', ['cold', 'warm']), ('prefix_cache_hits_verified', False),
    ('history_restoration_verified', False), ('native_kv_reads_and_writes', False),
    ('baseline_runtime_sha256', 'wrong'), ('policy_ids', ['wrong']),
    ('logit_max_abs', False), ('probability_max_abs', float('nan')),
])
def test_cache_speed_cannot_override_parity(field, value):
    baseline, candidate, parity = evidence()
    parity[field] = value
    with pytest.raises(ValueError):
        assess(baseline, candidate, parity, 'new')


def test_campaign_allows_input_and_engine_reuse_across_policies():
    baseline,candidate,parity=evidence()
    for row in (baseline,candidate):
        row['jobs']=row['jobs']+[dict(row['jobs'][0],job_id='point-2',policy_id='policy-2')]
        row['recordings']=row['completed_recordings']=18452
    candidate['stage_work_seconds']['engine_start']=.5
    parity['policy_ids']=['policy','policy-2']
    assert assess(baseline,candidate,parity,'new')['measured_speedup']==10


def test_campaign_cannot_drop_a_checkpoint_or_change_inputs():
    baseline,candidate,parity=evidence()
    candidate['jobs']=[dict(candidate['jobs'][0],policy_id='other')]
    with pytest.raises(ValueError,match='matched'):assess(baseline,candidate,parity,'new')


def test_hash_bound_evidence_and_missing_admission(tmp_path):
    def save(name, obj):
        raw = json.dumps(obj).encode(); path = tmp_path / name; path.write_bytes(raw)
        return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())
    proof = {k: save(k, v) for k, v in zip(('baseline', 'candidate', 'parity'), evidence())}
    config = dict(performance_admission=dict(evidence=save('proof', proof), runtime_sha256='new'))
    assert require_performance_admission(config)['status'] == 'performance_gate_passed'
    (tmp_path / 'candidate').write_text('{}')
    with pytest.raises(ValueError, match='hash changed'): require_performance_admission(config)
    with pytest.raises(ValueError, match='10x'): require_performance_admission({})


def test_entrypoints_reject_before_preparation_or_remote_dispatch():
    from tools.reports import run_committed_selection, run_multidevice_evaluation
    from tools.reports import context_full_transfer_operations, context_replica_transfer_operations
    for fn in (run_committed_selection.check, run_multidevice_evaluation.validate,
               context_full_transfer_operations.validate, context_replica_transfer_operations.validate):
        with pytest.raises(ValueError, match='10x'):
            fn({})

import copy
import json

import pytest

from test_context_efficiency_candidates import reports
from test_context_batched_service_candidate import engine_case
from tools.reports import run_context_kv_probe as probe


def case():
    _, rows, _ = engine_case()
    source = reports()
    config = dict(policy=dict(kv_cache_memory_bytes=2**30, max_batch_queries=32,
        max_batch_tokens=131072, snapshot='unchanged', policy_id=source['baseline']['policy_id']),
        kv_probe=dict(baseline_bytes=2**30, candidate_bytes=16*2**30))
    data = dict(baseline=source['baseline'], candidate=source['acoustic'])
    for arm, report in data.items():
        report['mode'] = 'acoustic'
        report['executed_policy'] = probe.arm_config(config, arm)['policy']
        for p in report['passes']:
            p['plan'] = dict(batch_queries=[len(rows)], batch_tokens=[sum(len(r['inputs']['input_ids'][0]) for r in rows)])
    return config, rows, data


def test_capacity_is_the_only_change_and_parent_is_not_mutated():
    config, _, _ = case(); original = copy.deepcopy(config)
    a, b = probe.arm_config(config, 'baseline'), probe.arm_config(config, 'candidate')
    assert b['policy'].pop('kv_cache_memory_bytes') == 16*2**30
    assert a['policy'].pop('kv_cache_memory_bytes') == 2**30
    assert a == b and config == original


def test_complete_comparison_survives_transport_json_round_trip():
    config, rows, data = case()
    result = probe.compare(data, rows, config)
    assert result == json.loads(json.dumps(result))
    assert result['speedup'] == 1
    assert result['exact_probabilities_and_logits'] and result['exact_sampling']
    assert result['training_dispatch_allowed'] is False
    assert set(result['arms']) == {'baseline', 'candidate'}


@pytest.mark.parametrize('fault', ['kv', 'path', 'mode', 'cache', 'coverage', 'order', 'missing'])
def test_wrong_runtime_or_incomplete_measurement_is_rejected(fault):
    config, rows, data = case(); r = data['candidate']
    if fault == 'kv': r['executed_policy']['kv_cache_memory_bytes'] = 2**30
    if fault == 'path': r['executed_policy']['snapshot'] = 'other'
    if fault == 'mode': r['mode'] = 'baseline'
    if fault == 'cache': r['passes'][1]['acoustic_counts']['conv'] = []
    if fault == 'coverage': r['passes'][1]['telemetry']['counts']['language'] = [[1]]
    if fault == 'order': r['passes'][3]['order'] = []
    if fault == 'missing': del data['baseline']
    with pytest.raises(ValueError): probe.compare(data, rows, config)


@pytest.mark.parametrize('value', [True, 0, 2**30, 0.5])
def test_invalid_capacity_rejected(value):
    config, _, _ = case(); config['kv_probe']['candidate_bytes'] = value
    with pytest.raises(ValueError): probe.arm_config(config, 'candidate')

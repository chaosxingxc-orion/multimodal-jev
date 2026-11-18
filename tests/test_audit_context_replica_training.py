import copy

import pytest

from multimodal_jev.context_topology_batch_plan import derive
from tools.reports.audit_context_replica_training import (check_result, check_schedule,
    expected_contract, mounted_names)


def batch_config(devices=6):
    return dict(training_devices=devices, tensor_parallel=1, pipeline_parallel=1,
                per_dp_replica_batch=64, microbatch=8, inference_instances=2,
                inference_batch_queries=48, source_queries=40513, epochs=2,
                efficiency_steps=100)


def candidate(devices=6):
    plan = derive(batch_config(devices))
    return dict(batch_config=batch_config(devices), physical_devices=list(range(devices)),
                global_batch=plan['global_batch'], microbatch=8,
                algorithm='dual_credit_grpo_floor_clipped_v1', dual_arm='joint',
                dual_runtime_identity={'profile': 'context-segmented-dual-logits-candidate-v1'},
                macro_client=dict(schema='jev-replica-macro-client-v1',
                                  max_queries=plan['macro_queries'], rollout_queries=32,
                                  replicas=[{}, {}],
                                  runtime_identity={'profile': 'context-segmented-dual-logits-candidate-v1'}),
                base_id='base', source_policy_id='source', source_contract={'token': 'jev'},
                runtime_profile='qwen3-asr-context-bf16-packed-row-v1', release_sha256='release',
                schedule_sha256='schedule', schedule_release_sha256='release', stage='task',
                behavior_bounds={'probability_max_abs': 2e-07})


def manifest(devices=6, services=2):
    return dict(training_devices=list(range(devices)),
                inference_services=[dict(name=chr(97+i), device=devices+i) for i in range(services)])


def test_contract_and_layout_derive_any_declared_width():
    c = candidate(); plan = derive(c['batch_config'])
    mounted = mounted_names(manifest())
    contract = expected_contract(c, mounted)
    assert contract['world_size'] == plan['dp_replicas'] == 6
    assert contract['macro_layout']['world_size'] == 6
    assert contract['macro_layout']['local_batch'] == 64
    assert contract['macro_layout']['global_batch'] == 384
    assert contract['macro_layout']['total_updates'] == 212
    assert contract['macro_layout']['accumulation'] == 8
    assert contract['schedule_release_sha256'] == 'release'
    assert mounted == [f'davinci{i}' for i in range(8)]


@pytest.mark.parametrize('change', ['devices', 'global', 'algorithm'])
def test_contract_rejects_inconsistent_declaration(change):
    c = candidate(); mounted = mounted_names(manifest())
    if change == 'devices':
        c['physical_devices'] = [0, 1, 2]
    elif change == 'global':
        c['global_batch'] = 128
    elif change == 'algorithm':
        c['algorithm'] = 'rloo_v1'
    with pytest.raises(ValueError):
        expected_contract(c, mounted)


def test_result_receipt_binds_declared_window():
    plan = derive(batch_config())
    good = dict(passed=True, start_update=0, stop_update=2, schedule_updates=212,
                learning_population=40513, engineering_prefix=True)
    check_result(good, 0, 2, plan)
    for key, value in [('schedule_updates', 634), ('learning_population', 40513-1),
                       ('engineering_prefix', False), ('stop_update', 3)]:
        bad = copy.deepcopy(good); bad[key] = value
        with pytest.raises(ValueError):
            check_result(bad, 0, 2, plan)


def test_formal_result_receipt_requires_the_declared_full_window():
    plan = derive(batch_config())
    good = dict(passed=True, start_update=0, stop_update=212, schedule_updates=212,
                learning_population=40513, engineering_prefix=False)
    check_result(good, 0, 212, plan, engineering=False)
    for key, value in [('engineering_prefix', True), ('stop_update', 2),
                       ('schedule_updates', 634)]:
        bad = copy.deepcopy(good); bad[key] = value
        with pytest.raises(ValueError):
            check_result(bad, 0, 212, plan, engineering=False)


def batches(plan):
    result = []
    for index in range(plan['total_updates']):
        epoch_end = (index + 1) % plan['updates_per_epoch'] == 0
        result.append(dict(real_queries=plan['tail_real_queries'] if epoch_end
                           else plan['global_batch']))
    return result


def test_tail_keeps_actual_real_denominator_and_exact_exposure():
    plan = derive(batch_config())
    check_schedule(batches(plan), plan)
    assert plan['updates_per_epoch'] == 106
    assert plan['tail_real_queries'] == 193 and plan['tail_padding_slots'] == 191
    assert sum(row['real_queries'] for row in batches(plan)) == 40513 * 2


@pytest.mark.parametrize('fault', ['tail_real', 'tail_padding', 'count', 'exposure'])
def test_schedule_denominator_faults_rejected(fault):
    plan = derive(batch_config()); rows = batches(plan)
    if fault == 'tail_real':
        rows[-1]['real_queries'] = 191
    elif fault == 'tail_padding':
        rows[-1]['real_queries'] = 194
    elif fault == 'count':
        rows.pop()
    elif fault == 'exposure':
        rows[0]['real_queries'] -= 1
    with pytest.raises(ValueError):
        check_schedule(rows, plan)

import pytest

from multimodal_jev.context_topology_batch_plan import derive
from tools.reports import context_replica_formal_candidate as formal
from tools.reports.context_replica_trainer_candidate import generate as replica_generate
from tools.reports.context_cached_reference_trainer import generate as cached_generate


def batch_config(world=6):
    return dict(training_devices=world, tensor_parallel=1, pipeline_parallel=1,
                per_dp_replica_batch=64, microbatch=8, inference_instances=2,
                inference_batch_queries=48, source_queries=40513, epochs=2,
                efficiency_steps=100)


CURVE = [0, 21, 42, 64, 85, 106, 127, 148, 170, 191, 212]


def declaration(world=6, steps=CURVE):
    plan = derive(batch_config(world))
    return dict(batch_config=batch_config(world), experiment_kind='formal_source',
                stage='task', start_update=0, stop_update=plan['total_updates'],
                engineering_prefix=False, training_dispatch_allowed=True,
                seed=317, evaluation_steps=list(steps),
                algorithm='dual_credit_grpo_floor_clipped_v1', dual_arm='joint')


def batches(plan):
    ids = [f'q{i}' for i in range(plan['configuration']['source_queries'])]
    result = []
    for epoch in range(plan['configuration']['epochs']):
        real = ids[epoch:] + ids[:epoch]; pos = 0
        for index in range(plan['updates_per_epoch']):
            count = (plan['tail_real_queries']
                     if index == plan['updates_per_epoch'] - 1 else plan['global_batch'])
            slots = [dict(id=real[pos + i], weight=1.0, padding=False)
                     for i in range(count)]
            pos += count
            slots += [dict(id=real[i % count], weight=0.0, padding=True)
                      for i in range(plan['global_batch'] - count)]
            result.append(dict(epoch=epoch, index=index, slots=slots, real_queries=count))
    return result


def test_declared_curve_must_span_zero_to_the_authenticated_endpoint():
    plan6 = derive(batch_config(6)); plan2 = derive(batch_config(2))
    assert plan6['total_updates'] == 212
    assert formal.window(plan6, CURVE) == CURVE
    assert formal.window(plan2, [0, *range(64, 634, 64), 634]) == [0, *range(64, 634, 64), 634]
    for bad in ([], [0], [21, 212], [0, 21, 21, 212], [0, 21, 211],
                [0, 21.0, 212], [0, 212, 211]):
        with pytest.raises(ValueError):
            formal.window(plan6, bad)


def test_contract_binds_world_batch_curve_and_denominators():
    c = declaration(6)
    plan = derive(c['batch_config'])
    result = formal.contract(c)
    assert result['experiment_kind'] == 'formal_source'
    assert result['updates'] == 212 and result['world_size'] == 6
    assert result['global_batch'] == 384 and result['updates_per_epoch'] == 106
    assert result['tail_real_queries'] == 193 and result['tail_padding_slots'] == 191
    assert result['evaluation_steps'] == c['evaluation_steps']
    assert result['recovery_checkpoint_steps'] == c['evaluation_steps'][1:]
    assert result['seed'] == 317
    assert plan['total_updates'] == result['updates']


@pytest.mark.parametrize('change', [
    'kind', 'start', 'stop', 'stage', 'resume', 'engineering', 'dispatch',
    'curve', 'endpoint', 'seed', 'arm'])
def test_contract_rejects_undeclared_windows(change):
    c = declaration(6)
    if change == 'kind':
        c['experiment_kind'] = 'efficiency'
    elif change == 'start':
        c['start_update'] = 1
    elif change == 'stop':
        c['stop_update'] = 212 - 1
    elif change == 'stage':
        c['stage'] = 'pre'
    elif change == 'resume':
        c['resume_root'] = '/remote/resume'
    elif change == 'engineering':
        c['engineering_prefix'] = True
    elif change == 'dispatch':
        c['training_dispatch_allowed'] = False
    elif change == 'curve':
        c['evaluation_steps'] = c['evaluation_steps'][:-1]
    elif change == 'endpoint':
        c['evaluation_steps'] = [*c['evaluation_steps'][:-1], 211]
    elif change == 'seed':
        c['seed'] = 7
    elif change == 'arm':
        c['dual_arm'] = 'reference'
    with pytest.raises(ValueError):
        formal.contract(c)


def test_complete_two_epoch_schedule_with_tails_is_accepted():
    plan = derive(batch_config(6)); rows = batches(plan)
    population = {slot['id'] for row in rows for slot in row['slots']}
    counts = formal.validate_schedule(rows, population, plan)
    assert len(counts) == 212
    assert counts[105] == 193 and counts[-1] == 193
    assert sum(counts) == 40513 * 2


@pytest.mark.parametrize('fault', [
    'tail_real', 'tail_padding', 'epoch_cursor', 'slot_count', 'padding_weight',
    'real_weight', 'unknown_id', 'missing_batch'])
def test_schedule_faults_are_rejected(fault):
    plan = derive(batch_config(6)); rows = batches(plan)
    population = {slot['id'] for row in rows for slot in row['slots']}
    if fault == 'tail_real':
        rows[105]['slots'][0]['padding'] = True
        rows[105]['slots'][0]['weight'] = 0.0
    elif fault == 'tail_padding':
        rows[105]['slots'].pop()
    elif fault == 'epoch_cursor':
        rows[3]['index'] = 4
    elif fault == 'slot_count':
        rows[0]['slots'].pop()
    elif fault == 'padding_weight':
        rows[105]['slots'][-1]['weight'] = 1.0
    elif fault == 'real_weight':
        rows[0]['slots'][0]['weight'] = 0.0
    elif fault == 'unknown_id':
        rows[0]['slots'][0]['id'] = 'absent'
    elif fault == 'missing_batch':
        rows.pop()
    with pytest.raises(ValueError):
        formal.validate_schedule(rows, population, plan)


def test_current_cached_formal_trainer_replaces_the_engineering_window():
    engineering = replica_generate('.')
    assert 'experiment_window(c)' in engineering
    # The current entry composes formal before the finite checkpoint rewrite.
    # Calling the historical formal wrapper directly applies those rewrites in
    # the wrong order; it is not the cached runtime's executable entry.
    source = cached_generate('.', formal=True)
    assert 'experiment_window(c)' not in source
    assert 'formal_contract(c)' in source
    assert "contract['formal_experiment']=formal" in source
    assert 'validate_schedule(batches,population.keys(),formal_plan(c))' in source
    assert "save_now=(index+1) in formal['recovery_checkpoint_steps']" in source
    assert 'save_now=True' not in source
    # The formal window carries no checkpoint cadence; the generated validation
    # must refuse that legacy field instead of reading it (v7 failed at runtime).
    assert "c['checkpoint_interval']" not in source
    assert "'checkpoint_interval' in c" in source
    # The admitted window replaces the engineering-only guard, not only the gate.
    assert 'engineering-only, not released' not in source
    assert "c['engineering_prefix'] is not False or c['training_dispatch_allowed'] is not True" in source
    compile(source, '<formal>', 'exec')

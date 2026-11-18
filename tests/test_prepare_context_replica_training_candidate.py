import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip('torch')

from multimodal_jev.context_topology_batch_plan import derive
from tools.reports.prepare_context_replica_training_candidate import prepare, topology


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def batch_config(world, services=2):
    return dict(training_devices=world, tensor_parallel=1, pipeline_parallel=1,
                per_dp_replica_batch=64, microbatch=8, inference_instances=services,
                inference_batch_queries=48, source_queries=40513, epochs=2,
                efficiency_steps=100)


def candidate(world, services=2):
    return dict(batch_config=batch_config(world, services),
                training_devices=list(range(world)),
                inference=[dict(name=chr(97 + i), device=world + i, max_batch_queries=48,
                                max_batch_tokens=16384) for i in range(services)],
                rollout=dict(rollout_queries=32, timeout_seconds=900.0),
                attempt_id=f'world-{world}-attempt')


@pytest.mark.parametrize('world,services,updates,global_batch', [
    (2, 2, 634, 128), (3, 1, 424, 192), (6, 2, 212, 384)])
def test_topology_derives_any_declared_width(world, services, updates, global_batch):
    plan, width, batch, declared, rollout = topology(candidate(world, services))
    assert plan == derive(batch_config(world, services))
    assert width == world and batch == global_batch
    assert plan['total_updates'] == updates
    assert [s['device'] for s in declared] == list(range(world, world + services))
    assert rollout == dict(rollout_queries=32, timeout_seconds=900.0)


@pytest.mark.parametrize('change', [
    'short-devices', 'duplicate-device', 'negative-device', 'shared-training-card',
    'wrong-service-count', 'wrong-batch-queries', 'duplicate-service-name',
    'bad-service-name', 'extra-service-key', 'wide-rollout', 'closed-timeout', 'bad-attempt'])
def test_topology_rejects_inconsistent_declarations(change):
    c = candidate(4, 2)
    if change == 'short-devices':
        c['training_devices'] = [0, 1, 2]
    elif change == 'duplicate-device':
        c['training_devices'] = [0, 1, 1, 3]
    elif change == 'negative-device':
        c['training_devices'] = [0, 1, 2, -1]
    elif change == 'shared-training-card':
        c['inference'][0]['device'] = 2
    elif change == 'wrong-service-count':
        c['inference'] = c['inference'][:1]
    elif change == 'wrong-batch-queries':
        c['inference'][0]['max_batch_queries'] = 32
    elif change == 'duplicate-service-name':
        c['inference'][1]['name'] = 'a'
    elif change == 'bad-service-name':
        c['inference'][1]['name'] = 'b-c'
    elif change == 'extra-service-key':
        c['inference'][1]['tensor_parallel'] = 1
    elif change == 'wide-rollout':
        c['rollout']['rollout_queries'] = 257
    elif change == 'closed-timeout':
        c['rollout']['timeout_seconds'] = 0
    elif change == 'bad-attempt':
        c['attempt_id'] = '../attempt'
    with pytest.raises(ValueError):
        topology(c)


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, sort_keys=True) + '\n')


def fixture(tmp_path, world=2, services=2):
    project = Path.cwd()
    parent = tmp_path / 'parent'
    for name in ('source/src/multimodal_jev', 'source/tools/jev', 'source/tools/reports'):
        (parent / name).mkdir(parents=True, exist_ok=True)
    config = dict(source_policy_id='policy-fixture', pins={}, remote_root='/remote/frozen',
                  source_policy='/remote/frozen/initial/policy',
                  encoded_audio_cache='/remote/frozen/cache',
                  audio_mirror_root='/remote/frozen/mirror',
                  schedule='/remote/frozen/source/schedule')
    # Modules inherited from the real parent package are represented by
    # placeholders, except the ones the packager ships from SOURCES.
    from tools.reports.prepare_context_replica_training_candidate import SOURCES, required_modules
    shipped = {Path(name).stem for name in SOURCES}
    for name in required_modules(project):
        if name not in shipped:
            write(parent / 'source/src/multimodal_jev'/f'{name}.py', dict(fixture=name))
    write(parent / 'config.json', config)
    write(parent / 'service-config.json', dict(policy=dict(snapshot='s0', adapter='a0')))
    files = {str(p.relative_to(parent)): sha(p) for p in parent.rglob('*') if p.is_file()}
    write(parent / 'PREPARED.json', dict(remote_root='/remote/frozen', files=files))
    c = candidate(world, services)
    schedule = tmp_path / 'schedule'; plan = derive(c['batch_config'])
    write(schedule / 'MANIFEST.json', dict(
        schema='jev-replica-source-schedule-v1', batch_plan=plan,
        total_updates=plan['total_updates'],
        total_real_exposures=c['batch_config']['source_queries'] * c['batch_config']['epochs']))
    parent_schedule = tmp_path / 'parent-schedule'
    write(parent_schedule / 'MANIFEST.json', dict(
        schema='context-full-learning-schedule-v1', release_sha256='release-fixture'))
    policy = tmp_path / 'policy'
    write(policy / 'processor/tokenizer.json', dict(fixture=1))
    write(policy / 'processor/tokenizer_config.json', dict(fixture=2))
    write(policy / 'MANIFEST.json', dict(policy_id='policy-fixture', files={
        name: sha(policy / name) for name in
        ('processor/tokenizer.json', 'processor/tokenizer_config.json')}))
    output = tmp_path / 'package'
    values = dict(parent=str(parent), parent_sha256=sha(parent / 'PREPARED.json'),
                  project=str(project), output=str(output), remote_root='/remote/replica',
                  schedule=str(schedule), schedule_sha256=sha(schedule / 'MANIFEST.json'),
                  parent_schedule=str(parent_schedule),
                  parent_schedule_sha256=sha(parent_schedule / 'MANIFEST.json'),
                  schedule_release_sha256='release-fixture', source_policy_id='policy-fixture',
                  local_policy=str(policy), **c)
    return values, output


@pytest.mark.parametrize('world,services', [(2, 2), (3, 1)])
def test_package_binds_declared_topology_end_to_end(tmp_path, world, services):
    values, output = fixture(tmp_path, world, services)
    result = prepare(values, {'workflow': 'test'})
    assert result['status'] == 'replica_integrated_prepared_only'
    assert result['batch_plan'] == derive(values['batch_config'])
    assert result['training_devices'] == list(range(world))
    config = json.loads((output / 'config.json').read_text())
    assert config['global_batch'] == 64 * world
    assert config['physical_devices'] == list(range(world))
    assert config['macro_client']['max_queries'] == 64 * world
    assert config['training_dispatch_allowed'] is False
    assert config['engineering_prefix'] is True
    assert 'experiment_kind' not in config and 'evaluation_steps' not in config
    assert result['experiment'] == dict(kind='engineering_prefix', updates=2)
    run = (output / 'run.py').read_text()
    assert f'--nproc-per-node={world}' in run
    expected = ', '.join(f"({chr(97 + i)!r}, {str(world + i)!r})" for i in range(services))
    assert expected in run
    assert repr(','.join(str(d) for d in range(world))) in run
    mounted = list(range(world + services))
    scope_hash = sha(output / 'source/tools/jev/context_configurable_device_scope.py')
    for i in range(services):
        service = json.loads((output / f'service-{chr(97 + i)}.json').read_text())
        scope = service['policy']['process_device_scope']
        assert service['policy']['physical_devices'] == [world + i]
        assert scope['mounted_physical'] == mounted
        assert scope['visible_ordinal'] == str(mounted.index(world + i))
        assert service['policy']['device_scope_module'] == dict(
            name='context_configurable_device_scope', sha256=scope_hash)


def test_formal_experiment_binds_the_derived_window(tmp_path):
    values, output = fixture(tmp_path, 6, 2)
    steps = [0, 21, 42, 64, 85, 106, 127, 148, 170, 191, 212]
    values['experiment'] = dict(kind='formal_source', evaluation_steps=steps, seed=317)
    result = prepare(values, {'workflow': 'test'})
    plan = derive(values['batch_config'])
    assert plan['total_updates'] == steps[-1]
    assert result['status'] == 'replica_formal_prepared_only'
    assert result['experiment']['evaluation_steps'] == steps
    assert result['experiment']['recovery_checkpoint_steps'] == steps[1:]
    config = json.loads((output / 'config.json').read_text())
    assert config['experiment_kind'] == 'formal_source' and config['stage'] == 'task'
    assert config['start_update'] == 0 and config['stop_update'] == plan['total_updates']
    assert config['engineering_prefix'] is False
    assert config['training_dispatch_allowed'] is True
    assert config['evaluation_steps'] == steps and config['seed'] == 317
    assert 'checkpoint_interval' not in config
    run = (output / 'run.py').read_text()
    assert 'context_replica_formal_candidate.py' in run and 'formal_contract(c)' in run
    assert (output / 'source/tools/reports/context_replica_formal_candidate.py').is_file()
    assert 'tools/reports/context_replica_formal_candidate.py' in json.loads(
        (output / 'RUNTIME-SOURCES.json').read_text())


@pytest.mark.parametrize('experiment', [
    dict(kind='efficiency', evaluation_steps=[0, 212], seed=317),
    dict(kind='formal_source', evaluation_steps=[0, 191], seed=317),
    dict(kind='formal_source', evaluation_steps=[0, 106, 106, 212], seed=317),
    dict(kind='formal_source', evaluation_steps=[0, 212], seed=7),
    dict(kind='formal_source', evaluation_steps=[0, 212]),
    dict(kind='formal_source', evaluation_steps=[0, 212], seed=317, stop_update=2)])
def test_formal_experiment_rejects_undeclared_windows(tmp_path, experiment):
    values, _ = fixture(tmp_path)
    values['experiment'] = experiment
    with pytest.raises(ValueError):
        prepare(values, {'workflow': 'test'})


def test_package_rejects_referenced_module_missing_from_source(tmp_path, monkeypatch):
    import tools.reports.prepare_context_replica_training_candidate as packager
    values, output = fixture(tmp_path)
    monkeypatch.setattr(packager, 'required_modules', lambda root: ['definitely_missing_module'])
    with pytest.raises(ValueError, match='unpackaged modules: definitely_missing_module'):
        packager.prepare(values, {'workflow': 'test'})

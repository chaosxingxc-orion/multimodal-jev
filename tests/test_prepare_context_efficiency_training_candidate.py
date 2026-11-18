import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip('torch')

from multimodal_jev.context_topology_batch_plan import derive
from tools.reports.prepare_context_efficiency_training_candidate import (
    EFFICIENCY_BATCH_QUERIES, EFFICIENCY_BATCH_TOKENS, RUNTIME, declaration, prepare, topology)

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / 'configs/experiments/context-efficiency-v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def design():
    return json.loads(DESIGN.read_text())


def candidate():
    return declaration(design())


def test_packager_uses_the_efficiency_owner_not_the_batched_one():
    import tools.reports.context_efficiency_owner_candidate as efficiency
    import tools.reports.prepare_context_efficiency_training_candidate as packager
    assert packager.transformed_source is efficiency.transformed_source
    assert packager.RUNTIME == efficiency.RUNTIME == 'context-efficiency-candidate-v1'
    assert (EFFICIENCY_BATCH_QUERIES, EFFICIENCY_BATCH_TOKENS) == (48, 65536)


def test_design_config_renders_dp2_topology():
    c = candidate()
    plan, world, batch, services, rollout = topology(c)
    assert world == 2 and plan['dp_replicas'] == 2
    assert batch == plan['global_batch'] == 128
    assert plan['total_updates'] == 634
    assert c['batch_config']['training_devices'] == 2
    assert c['batch_config']['per_dp_replica_batch'] == 64
    assert c['batch_config']['efficiency_steps'] == 100
    assert c['training_devices'] == [0, 1]
    assert [s['device'] for s in services] == [2, 3]
    assert [s['name'] for s in services] == ['a', 'b']
    for service in services:
        assert service['max_batch_queries'] == EFFICIENCY_BATCH_QUERIES
        assert service['max_batch_tokens'] == EFFICIENCY_BATCH_TOKENS
    assert rollout == dict(rollout_queries=32, timeout_seconds=900.0)


@pytest.mark.parametrize('world', [2, 3, 6])
def test_topology_is_read_from_the_design_config_not_hardcoded(world):
    d = design()
    devices = dict(training_devices=world, physical_devices_train=list(range(world)),
                   physical_devices_inference=[world, world + 1], inference_instances=2)
    d['topology'].update(devices)
    d['efficiency_service'].update(devices)
    plan, width, batch, services, _ = topology(declaration(d))
    assert width == plan['dp_replicas'] == world
    assert batch == 64 * world
    assert [s['device'] for s in services] == [world, world + 1]


@pytest.mark.parametrize('change', [
    'tokens-low', 'tokens-high', 'queries-32', 'queries-64'])
def test_topology_rejects_non_efficiency_budget(change):
    c = candidate()
    if change == 'tokens-low':
        c['inference'][0]['max_batch_tokens'] = 16384
    elif change == 'tokens-high':
        c['inference'][1]['max_batch_tokens'] = 131072
    elif change == 'queries-32':
        c['inference'][0]['max_batch_queries'] = 32
        c['batch_config']['inference_batch_queries'] = 32
    elif change == 'queries-64':
        c['inference'] = [dict(s, max_batch_queries=64) for s in c['inference']]
        c['batch_config']['inference_batch_queries'] = 64
    # The batch-plan derivation or the packager's explicit budget gate rejects
    # every non-48/65536 declaration.
    with pytest.raises(ValueError):
        topology(c)


@pytest.mark.parametrize('change', ['train-count', 'infer-count', 'service-mismatch'])
def test_declaration_rejects_inconsistent_design(change):
    d = design()
    if change == 'train-count':
        d['topology']['training_devices'] = 3
    elif change == 'infer-count':
        d['efficiency_service']['inference_instances'] = 1
    elif change == 'service-mismatch':
        d['efficiency_service']['physical_devices_inference'] = [2, 4]
    with pytest.raises(ValueError):
        declaration(d)


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, sort_keys=True) + '\n')


def fixture(tmp_path):
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
    from tools.reports.prepare_context_efficiency_training_candidate import SOURCES, required_modules
    shipped = {Path(name).stem for name in SOURCES}
    for name in required_modules(project):
        if name not in shipped:
            write(parent / 'source/src/multimodal_jev'/f'{name}.py', dict(fixture=name))
    write(parent / 'config.json', config)
    write(parent / 'service-config.json', dict(policy=dict(snapshot='s0', adapter='a0')))
    files = {str(p.relative_to(parent)): sha(p) for p in parent.rglob('*') if p.is_file()}
    write(parent / 'PREPARED.json', dict(remote_root='/remote/frozen', files=files))
    c = candidate()
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


def test_package_binds_efficiency_owner_and_dp2_budget(tmp_path):
    values, output = fixture(tmp_path)
    result = prepare(values, {'workflow': 'test'})
    assert result['status'] == 'replica_integrated_prepared_only'
    assert result['dual_runtime_identity']['profile'] == RUNTIME
    assert result['training_devices'] == [0, 1]
    config = json.loads((output / 'config.json').read_text())
    assert config['dual_runtime_identity']['profile'] == RUNTIME
    assert config['global_batch'] == 128
    assert config['physical_devices'] == [0, 1]
    assert config['training_dispatch_allowed'] is False
    for name, device in (('a', 2), ('b', 3)):
        policy = json.loads((output / f'service-{name}.json').read_text())['policy']
        assert policy['candidate_runtime'] == RUNTIME
        assert policy['physical_devices'] == [device]
        assert policy['max_batch_queries'] == EFFICIENCY_BATCH_QUERIES
        assert policy['max_batch_tokens'] == EFFICIENCY_BATCH_TOKENS
    copied = output / 'source/tools/reports/context_efficiency_owner_candidate.py'
    assert copied.is_file()
    assert 'gpu_memory_utilization=.6' in copied.read_text()
    # The inherited batched owner still ships, but the packager transform is the
    # efficiency one (asserted above); the frozen switch source stays untouched.
    assert (output / 'source/tools/reports/context_batched_owner_candidate.py').is_file()
    frozen = output / 'source/tools/jev/context_switch_rollout.py'
    assert 'gpu_memory_utilization=.4' in frozen.read_text()


def test_generated_files_compile(tmp_path):
    values, output = fixture(tmp_path)
    prepare(values, {'workflow': 'test'})
    for path in output.rglob('*.py'):
        compile(path.read_text(), f'<{path.name}>', 'exec')
    assert (output / 'PREPARED.json').is_file()


def test_existing_output_directory_rejected(tmp_path):
    values, output = fixture(tmp_path)
    output.mkdir(parents=True)
    marker = output / 'keep.txt'
    marker.write_text('untouched\n')
    with pytest.raises(ValueError, match='new isolated candidate paths required'):
        prepare(values, {'workflow': 'test'})
    assert marker.read_text() == 'untouched\n'

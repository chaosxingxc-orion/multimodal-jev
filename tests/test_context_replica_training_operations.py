from pathlib import Path
import re

import pytest

from multimodal_jev.context_topology_batch_plan import derive
from tools.reports.context_replica_training_operations import command, experiment_window, path


REMOTE = '/data/host/experiments/replica-v1'


def topology(world, services):
    names = [chr(97 + i) for i in range(services)]
    m = dict(remote_root=REMOTE,
             training_devices=list(range(world)),
             inference_services=[dict(name=n, device=world + i)
                                 for i, n in enumerate(names)],
             layout=dict(configs=['config.json', 'run.py', 'RUNTIME-SOURCES.json',
                                  'PREPARED.json'],
                         services={n: f'service-{n}.json' for n in names}))
    config = dict(source_policy='/data/host/experiments/base/initial/policy',
                  encoded_audio_cache='/data/host/experiments/cache/encoded',
                  audio_mirror_root='/data/host/experiments/mirror',
                  pins={REMOTE + '/source/x.py': 'hash',
                        REMOTE + '/service-a.json': 'hash',
                        REMOTE + '/config.json': 'hash',
                        REMOTE + '/run.py': 'hash',
                        REMOTE + '/RUNTIME-SOURCES.json': 'hash',
                        REMOTE + '/PREPARED.json': 'hash',
                        REMOTE + '/source/schedule/MANIFEST.json': 'hash'})
    c = dict(tag='replica-topology-v1', shared_root='/mnt/shared',
             driver_root='/usr/local/Ascend/driver')
    return c, m, config


@pytest.mark.parametrize('world,services', [(2, 2), (3, 1), (6, 2)])
def test_devices_and_mounts_follow_declared_topology(world, services):
    argv = command(*topology(world, services))
    devices = [argv[i + 1] for i, v in enumerate(argv) if v == '--device']
    declared = sorted(range(world + services))
    npu = [p for p in devices if re.fullmatch(r'/dev/davinci\d+:/dev/davinci\d+', p)]
    assert npu == [f'/dev/davinci{d}:/dev/davinci{d}' for d in declared]
    assert len(devices) == len(declared) + 3
    binds = [argv[i + 1] for i, v in enumerate(argv) if v == '-v']
    for top in ('config.json', 'run.py', 'RUNTIME-SOURCES.json', 'PREPARED.json',
                *[f'service-{chr(97 + i)}.json' for i in range(services)], 'source'):
        assert f'{REMOTE}/{top}:{REMOTE}/{top}:ro' in binds
    assert f'{REMOTE}/exchange:/jev-exchange' in binds
    assert argv[-1] == REMOTE + '/run.py' and argv[-2] == 'python'
    entry = argv.index('--entrypoint')
    assert argv[entry + 1] == '/opt/multimodal-jev/runtime-entrypoint.sh'
    env = dict(item.split('=', 1) for item in
               [argv[i + 1] for i, v in enumerate(argv) if v == '-e'])
    assert env['PYTHONPATH'].split(':')[0] == REMOTE + '/source'


def test_dependency_outside_declared_mounts_rejected():
    c, m, config = topology(2, 2)
    config['pins']['/data/unmounted/x.py'] = 'hash'
    with pytest.raises(ValueError, match='outside explicit mounts'):
        command(c, m, config)


def test_runtime_root_must_not_contain_reusable_inputs():
    c, m, config = topology(2, 2)
    c['shared_root'] = REMOTE + '/shared'
    with pytest.raises(ValueError, match='disjoint'):
        command(c, m, config)


@pytest.mark.parametrize('value', ['relative/path', '/a/../b', '/', '/a//b', '', None])
def test_runtime_paths_must_be_canonical_and_absolute(value):
    with pytest.raises(ValueError):
        path(value)


def formal_config(world=6, steps=(0, 53, 106, 159, 212)):
    batch = dict(training_devices=world, tensor_parallel=1, pipeline_parallel=1,
                 per_dp_replica_batch=64, microbatch=8, inference_instances=2,
                 inference_batch_queries=48, source_queries=40513, epochs=2,
                 efficiency_steps=100)
    plan = derive(batch)
    return dict(batch_config=batch, experiment_kind='formal_source', stage='task',
                start_update=0, stop_update=plan['total_updates'],
                engineering_prefix=False, training_dispatch_allowed=True,
                seed=317, evaluation_steps=list(steps),
                algorithm='dual_credit_grpo_floor_clipped_v1', dual_arm='joint'), plan


def test_engineering_window_is_the_admitted_prefix():
    plan = derive(dict(training_devices=6, tensor_parallel=1, pipeline_parallel=1,
                       per_dp_replica_batch=64, microbatch=8, inference_instances=2,
                       inference_batch_queries=48, source_queries=40513, epochs=2,
                       efficiency_steps=100))
    c = dict(engineering_prefix=True, training_dispatch_allowed=False,
             start_update=0, stop_update=2, checkpoint_interval=1)
    assert experiment_window(c, plan) == 'engineering_prefix'
    for key, value in [('stop_update', 3), ('checkpoint_interval', 2),
                       ('training_dispatch_allowed', True), ('engineering_prefix', False)]:
        bad = dict(c); bad[key] = value
        with pytest.raises(ValueError):
            experiment_window(bad, plan)


def test_declared_formal_window_is_accepted_without_per_width_constants():
    c, plan = formal_config(6)
    assert plan['total_updates'] == 212
    assert experiment_window(c, plan) == 'formal_source'
    c2, plan2 = formal_config(2, (0, 317, 634))
    assert plan2['total_updates'] == 634
    assert experiment_window(c2, plan2) == 'formal_source'


@pytest.mark.parametrize('change', ['stop', 'curve', 'dispatch', 'engineering', 'resume'])
def test_undeclared_windows_are_rejected(change):
    c, plan = formal_config(6)
    if change == 'stop':
        c['stop_update'] = 2
    elif change == 'curve':
        c['evaluation_steps'] = [0, 106, 106, 212]
    elif change == 'dispatch':
        c['training_dispatch_allowed'] = False
    elif change == 'engineering':
        c['engineering_prefix'] = True
    elif change == 'resume':
        c['resume_root'] = '/remote/resume'
    with pytest.raises(ValueError):
        experiment_window(c, plan)


def release_fixture(tmp_path, destination='/home/builder/project'):
    import hashlib
    import json
    release = tmp_path / 'release'
    root = tmp_path / 'evidence'
    document = root / 'configs/reviews/audit.json'
    document.parent.mkdir(parents=True)
    document.write_text('{"review": 1}\n')
    digest = hashlib.sha256(document.read_bytes()).hexdigest()
    (release / 'publication-config.json').parent.mkdir(parents=True, exist_ok=True)
    (release / 'publication-config.json').write_text(json.dumps(
        dict(pins={destination + '/configs/reviews/audit.json': digest,
                   '/mnt/shared/dataset/MANIFEST.json': 'shared'})))
    return str(release), str(root), destination, document, hashlib


def test_declared_release_evidence_mounts_at_recorded_absolute_paths(tmp_path):
    release, root, destination, document, _ = release_fixture(tmp_path)
    c, m, config = topology(2, 2)
    config['release'] = release
    c['release_evidence'] = dict(root=root, destination=destination)
    argv = command(c, m, config)
    binds = [argv[i + 1] for i, v in enumerate(argv) if v == '-v']
    assert f'{root}:{destination}:ro' in binds


def test_release_evidence_must_match_the_release_pins(tmp_path):
    release, root, destination, document, hashlib = release_fixture(tmp_path)
    document.write_text('tampered\n')
    c, m, config = topology(2, 2)
    config['release'] = release
    c['release_evidence'] = dict(root=root, destination=destination)
    with pytest.raises(ValueError, match='relocated release evidence differs'):
        command(c, m, config)


def test_release_evidence_destination_must_be_disjoint_from_the_run_root(tmp_path):
    release, root, destination, _, _ = release_fixture(tmp_path, REMOTE + '/evidence')
    c, m, config = topology(2, 2)
    config['release'] = release
    c['release_evidence'] = dict(root=root, destination=destination)
    with pytest.raises(ValueError, match='disjoint'):
        command(c, m, config)

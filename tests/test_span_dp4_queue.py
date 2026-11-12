import importlib.util
import json
from pathlib import Path
import subprocess

import pytest
from multimodal_jev.baseline import file_hash


@pytest.fixture
def queue():
    p = Path(__file__).resolve().parents[1]/'tools/jev/queue_span_dp4_resume.py'
    spec = importlib.util.spec_from_file_location('dp4_queue', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize('fault', [None, 'training_running', 'evaluation_running', 'evaluation_failed',
                                  'queue_failed', 'missing_dispatch', 'changed_launch', 'wrong_target'])
def test_waits_for_future_evaluation_not_just_training(queue, tmp_path, monkeypatch, fault):
    root = tmp_path/'queue'; evaluation = tmp_path/'evaluation'
    write(root/'launch.json', {'pid': 0, 'argv': []})
    write(root/'run/terminal.json', {'exit_code': 1 if fault == 'queue_failed' else 0})
    p = write(evaluation/'launch.json', {'container_id': 'eval', 'launch_exit_code': 0})
    if fault != 'missing_dispatch':
        write(root/'run/dispatched.json', {'evaluation': 'wrong' if fault == 'wrong_target' else str(evaluation),
              'container_id': 'eval', 'launch_sha256': file_hash(p)})
    if fault == 'changed_launch': p.write_text('{}')
    config = {'containers': ['train'], 'evaluation_queues': [{'root': str(root), 'evaluation': str(evaluation), 'pins': {}}]}
    rows = {cid: {'State': {'Status': 'running' if fault == name else 'exited',
                          'ExitCode': int(fault == 'evaluation_failed' and cid == 'eval')}}
            for cid, name in [('train', 'training_running'), ('eval', 'evaluation_running')]}
    monkeypatch.setattr(queue, 'inspect', lambda ids: {k: rows[k] for k in ids})
    if fault in (None, 'training_running', 'evaluation_running'):
        assert queue.dependencies(config) == ([] if fault is None else ['train' if fault == 'training_running' else 'eval'])
    else:
        with pytest.raises(ValueError): queue.dependencies(config)


@pytest.mark.parametrize('fault', [None, 'reservation4', 'reservation7', 'privileged', 'host_busy', 'host_denied', 'transport'])
def test_all_four_cards_checked_including_host_processes(queue, monkeypatch, fault):
    mapped = '/dev/davinci4' if fault == 'reservation4' else '/dev/davinci7' if fault == 'reservation7' else '/dev/davinci0'
    monkeypatch.setattr(queue, 'inspect', lambda ids: {'other': {'Id': 'other', 'HostConfig': {
        'Devices': [{'PathOnHost': mapped}], 'Privileged': fault == 'privileged'}}})
    def remote(argv):
        if argv[1] == 'ps': return subprocess.CompletedProcess(argv, 0, 'other\n', '')
        return subprocess.CompletedProcess(argv, 255 if fault == 'transport' else 0,
            json.dumps({'busy': [1] if fault == 'host_busy' else [], 'denied': [1] if fault == 'host_denied' else []}), '')
    monkeypatch.setattr(queue, 'remote', remote)
    if fault:
        with pytest.raises((ValueError, subprocess.CalledProcessError)): queue.check_devices('image')
    else: queue.check_devices('image')


@pytest.mark.parametrize('fault', [None, 'pending', 'busy', 'create', 'timeout', 'start'])
def test_receipt_precedes_start_no_recreation(queue, tmp_path, monkeypatch, fault):
    root = tmp_path/'training'; root.mkdir(); output = tmp_path/'out'; output.mkdir()
    plan = dict(root=str(root), image='image', physical_devices=[4,5,6,7],
                config_sha256='config', source_manifest_sha256='source')
    monkeypatch.setattr(queue, 'prepared_plan', lambda c: (plan, ['docker','create','fixed-name']))
    monkeypatch.setattr(queue, 'dependencies', lambda c: ['pending'] if fault == 'pending' else [])
    def devices(image):
        if fault == 'busy': raise ValueError('busy')
    monkeypatch.setattr(queue, 'check_devices', devices)
    cid = 'a'*64; calls = []
    def remote(argv):
        calls.append(argv)
        assert (output/'create-attempt.json').exists()
        if argv[1] == 'start':
            assert json.loads((root/'launch.json').read_text())['container_id'] == cid
        if fault == 'timeout': raise subprocess.TimeoutExpired(argv, 60)
        return subprocess.CompletedProcess(argv, int(fault == argv[1]), cid+'\n', '')
    monkeypatch.setattr(queue, 'remote', remote)
    monkeypatch.setattr(queue, 'inspect', lambda ids: {cid: {'Image': 'image', 'State': {'Status':'running'}}})
    c = {'lock_directory': str(tmp_path)}
    if fault:
        with pytest.raises((ValueError, subprocess.SubprocessError)): queue.dispatch(c, output)
        assert not (output/'dispatched.json').exists()
        assert sum(a[1] == 'create' for a in calls) <= 1
        if fault in ('busy', 'pending'): assert not calls
        if fault == 'timeout': assert (output/'create-attempt.json').exists()
    else:
        queue.dispatch(c, output)
        assert [a[1] for a in calls] == ['create','start']
        assert (output/'dispatched.json').exists()


@pytest.mark.parametrize('fault', [None, 'claimed', 'wrong_card', 'changed_config', 'changed_source', 'cpu_failed'])
def test_prepared_plan_checks_real_assets_and_topology(queue, tmp_path, fault):
    root = tmp_path/'prepared'; root.mkdir()
    source = root/'source'; source.mkdir(); (source/'test.py').write_text('pass\n')
    manifest = write(root/'source-manifest.json', {'test.py': file_hash(source/'test.py')})
    config = write(root/'config.json', {'pins': {str(source/'test.py'): file_hash(source/'test.py')}})
    argv = ['docker','run','-d']
    for dev in ('davinci4','davinci5','davinci6','davinci7','davinci_manager','devmm_svm','hisi_hdc'):
        argv += ['--device', f'/dev/{dev}:/dev/{dev}']
    argv += ['image','torchrun']
    plan = write(tmp_path/'plan.json', {'root': str(root), 'status': 'prepared_not_launched',
        'physical_devices': [4,5,6,7], 'cpu_audit': {'passed': fault != 'cpu_failed'},
        'config_sha256': file_hash(config), 'source_manifest_sha256': file_hash(manifest),
        'argv': argv, 'image': 'image'})
    if fault == 'claimed': (root/'run').mkdir()
    if fault == 'wrong_card':
        d = json.loads(plan.read_text()); d['argv'][4] = '/dev/davinci0:/dev/davinci0'; write(plan, d)
    if fault == 'changed_config': config.write_text('{}')
    if fault == 'changed_source': (source/'test.py').write_text('changed')
    c = {'pins': {str(plan): file_hash(plan)}, 'prepared_plan': str(plan)}
    if fault:
        with pytest.raises(ValueError): queue.prepared_plan(c)
    else:
        p, create = queue.prepared_plan(c)
        assert create == ['docker','create',*argv[3:]]

"""Independent stage boundary audit uses real four-rank tensor files."""
import copy
import json
from pathlib import Path

import pytest
torch = pytest.importorskip('torch')
from multimodal_jev.baseline import file_hash
from multimodal_jev.resident_service_admission import IMAGE
from multimodal_jev import simple_stage2_audit as module
from multimodal_jev.simple_resident_selection import SELECTION_UPDATES
from test_simple_sampled_checkpoint import fixture as tensor_fixture


def fixture(tmp_path, monkeypatch):
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))
    root = tmp_path / 'train'
    train = dict(schedule_sha256='schedule', stage2_initialization=dict(stop_update=634,
        selection_updates=list(SELECTION_UPDATES)))
    put(root / 'config.json', train)
    put(root / 'launch.json', dict(image=IMAGE, container_id='training',
        config_sha256=file_hash(root / 'config.json')))
    observed = tmp_path / 'observed.json'
    put(observed, dict(image=IMAGE, container_id='training', state=dict(Status='exited', ExitCode=0)))
    put(root / 'terminal.json', dict(exit_code=0))
    states, _, _ = tensor_fixture()
    paths = []
    for rank, state in enumerate(states):
        state['updates'] = 634
        for value in state['optimizer']['state'].values(): value['step'].fill_(634)
        before = tmp_path / f'source-{rank}.pt'; torch.save(state, before); paths.append(str(before))
        initial = copy.deepcopy(state)
        initial.update(updates=0, next_update=0, stage='simple_stage2_initial',
            reference_policy_id='base:final', adapter_id='final', schedule_sha256='schedule')
        initial['optimizer']['state'] = {}
        out = root / 'run' / f'rank-{rank}'; out.mkdir(parents=True)
        torch.save(initial, out / 'stage2-initial.pt')
        put(out / 'terminal.json', dict(exit_code=0))
        put(out / 'stage2-execution.json', dict(execution_dtypes={str(i): ['torch.bfloat16']*2 for i in range(112)}))
    initialized = dict(reference_policy_id='base:final', shard_paths=paths,
        files={p: file_hash(p) for p in paths})
    # Final Stage1 authentication and full parity have separate integration
    # tests; this fixture covers process/source boundary and saved tensor audit.
    monkeypatch.setattr(module, 'authenticate', lambda *a, **k: initialized)
    config = dict(training_root=str(root), training_observed=str(observed),
        pins={str(p): file_hash(p) for p in [root / 'config.json', root / 'launch.json', observed]})
    return config, root, put


def test_independent_four_rank_initial_masters_and_empty_optimizer(tmp_path, monkeypatch):
    config, _, _ = fixture(tmp_path, monkeypatch)
    result = module.source(config, verify_parity=None)
    assert len(result['checks']) == 4 and len(result['names']) == 112
    assert all(c['exact_masters'] and c['optimizer_reset'] for c in result['checks'])


def test_independent_source_audit_refuses_npu_visible_process(tmp_path, monkeypatch):
    config, _, _ = fixture(tmp_path, monkeypatch)
    original = Path.glob
    monkeypatch.setattr(Path, 'glob', lambda path, pattern: iter([Path('/dev/davinci0')])
        if path == Path('/dev') else original(path, pattern))
    with pytest.raises(ValueError, match='CPU-only'):
        module.source(config, verify_parity=None)


def stopped_fixture(tmp_path, monkeypatch):
    config, root, put = fixture(tmp_path, monkeypatch)
    config.update(audit_mode='stopped_prefix', completed_updates=64)
    observed = Path(config['training_observed'])
    value = json.loads(observed.read_text()); value['state']['ExitCode'] = 137
    put(observed, value); config['pins'][str(observed)] = file_hash(observed)
    (root / 'terminal.json').unlink()
    for rank in range(4):
        out = root / 'run' / f'rank-{rank}'
        execution = json.loads((out / 'stage2-execution.json').read_text())
        put(root / 'run' / f'loop-rank-{rank}/execution-0064.json',
            dict(execution, rank=rank, updates=64))
        (out / 'terminal.json').unlink(); (out / 'stage2-execution.json').unlink()
    return config, root, put


def test_crashed_container_prefix_requires_exact_initial_state_and_checkpoint_hooks(tmp_path, monkeypatch):
    config, _, _ = stopped_fixture(tmp_path, monkeypatch)
    result = module.source(config, verify_parity=None)
    assert result['stopped_prefix'] and result['completed_updates'] == 64
    assert len(result['checks']) == 4
    assert sum('execution-0064.json' in p for p in result['files']) == 4


@pytest.mark.parametrize('fault', ['running', 'missing_exit', 'wrong_step', 'wrong_rank', 'fp32', 'no_hooks'])
def test_prefix_rejects_live_or_missing_execution_evidence(tmp_path, monkeypatch, fault):
    config, root, put = stopped_fixture(tmp_path, monkeypatch)
    path = root / 'run/loop-rank-2/execution-0064.json'
    if fault == 'no_hooks': path.unlink()
    elif fault in ('running', 'missing_exit'):
        path = Path(config['training_observed']); value = json.loads(path.read_text())
        if fault == 'running': value['state']['Status'] = 'running'
        else: value['state'].pop('ExitCode')
        put(path, value); config['pins'][str(path)] = file_hash(path)
    else:
        value = json.loads(path.read_text())
        if fault == 'wrong_step': value['updates'] = 63
        elif fault == 'wrong_rank': value['rank'] = 1
        else: value['execution_dtypes']['0'] = ['torch.float32'] * 2
        put(path, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        module.source(config, verify_parity=None)


@pytest.mark.parametrize('config', [dict(audit_mode='unknown'), dict(completed_updates=634),
    *[dict(audit_mode='stopped_prefix', completed_updates=v) for v in (True, 0, 634, 635, 1.5, None)]])
def test_audit_boundary_cannot_silently_reduce_or_promote_scope(config):
    with pytest.raises(ValueError): module.audit_boundary(config)


def test_audit_boundary_keeps_terminal_default_and_explicit_prefix():
    assert module.audit_boundary({}) == (634, False)
    assert module.audit_boundary(dict(audit_mode='stopped_prefix', completed_updates=317)) == (317, True)


@pytest.mark.parametrize('fault', ['live', 'wrong_image', 'unpin', 'rank_failed', 'execution',
    'master', 'master_dtype', 'adam', 'cursor', 'reference', 'layout', 'schedule', 'short_run'])
def test_reject_unaccepted_process_and_initial_state(tmp_path, monkeypatch, fault):
    config, root, put = fixture(tmp_path, monkeypatch)
    out = root / 'run/rank-2'
    if fault in ('master', 'master_dtype', 'adam', 'cursor', 'reference', 'layout', 'schedule'):
        path = out / 'stage2-initial.pt'; value = torch.load(path, weights_only=True)
        if fault == 'master': value['parameters'][value['names'][0]] += 1e-6
        elif fault == 'master_dtype': value['parameters'][value['names'][0]] = value['parameters'][value['names'][0]].bfloat16()
        elif fault == 'adam': value['optimizer']['state'] = {0: {}}
        elif fault == 'cursor': value['updates'] = 634
        elif fault == 'reference': value['reference_policy_id'] = 'base:other'
        elif fault == 'layout': value['names'] = list(reversed(value['names']))
        else: value['schedule_sha256'] = 'other'
        torch.save(value, path)
    elif fault == 'unpin': config['pins'].pop(config['training_observed'])
    else:
        if fault in ('live', 'wrong_image'): path = Path(config['training_observed'])
        elif fault == 'short_run': path = root / 'config.json'
        else: path = out / ('terminal.json' if fault == 'rank_failed' else 'stage2-execution.json')
        value = json.loads(path.read_text())
        if fault == 'live': value['state']['Status'] = 'running'
        elif fault == 'wrong_image': value['image'] = 'other'
        elif fault == 'short_run': value['stage2_initialization']['stop_update'] = 3
        elif fault == 'rank_failed': value['exit_code'] = 1
        else: value['execution_dtypes']['0'] = ['torch.float32']*2
        put(path, value)
        if str(path) in config['pins']: config['pins'][str(path)] = file_hash(path)
        if fault == 'short_run':
            launch_path = root / 'launch.json'; launch = json.loads(launch_path.read_text())
            launch['config_sha256'] = file_hash(path); put(launch_path, launch)
            config['pins'][str(launch_path)] = file_hash(launch_path)
    with pytest.raises(ValueError): module.source(config, verify_parity=None)


def selection_fixture(tmp_path, monkeypatch):
    """Audit orchestration only; full9226 raw scoring is tested separately."""
    import multimodal_jev.simple_resident_selection as selection
    import multimodal_jev.slurp_official_training as release
    monkeypatch.setattr(release, 'load', lambda *a, **k: ([], {}, {}))
    metrics = dict(accuracy=.2, macro_F1_fixed60=.1, macro_recall_supported=.15, invalid_rate=0.)
    policies = {step: 'base:adapter'+str(step) for step in SELECTION_UPDATES}
    calls = []
    def verify(path, digest, *args, **kwargs):
        calls.append(path)
        step = int(path.parent.name.split('-')[1])
        return dict(updates=step, policy_id=policies[step], metrics=metrics)
    monkeypatch.setattr(selection, 'verify_result', verify)
    for step in SELECTION_UPDATES:
        path = tmp_path / 'run/evaluation' / f'selection-{step:04d}/result.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(files={}, checkpoint=dict(files={}), inputs='input', inputs_sha256='hash')))
        for rank in range(4):
            binding = tmp_path / 'run' / f'loop-rank-{rank}' / f'selection-{step:04d}.json'
            binding.parent.mkdir(exist_ok=True)
            binding.write_text(json.dumps(dict(passed=True, updates=step, policy_id=policies[step],
                recordings=9226, result=str(path), result_sha256=file_hash(path), metrics=metrics)))
    return dict(release='release', release_sha256='hash'), policies, calls


def test_all_eleven_selection_points_bound_to_exact_trained_policies(tmp_path, monkeypatch):
    train, policies, calls = selection_fixture(tmp_path, monkeypatch)
    files = {}
    result = module.selections(tmp_path, train, policies, files)
    assert len(result) == len(calls) == 11 and len(files) == 56


def test_prefix_checks_all_due_selections_and_excludes_failed_suffix(tmp_path, monkeypatch):
    train, policies, calls = selection_fixture(tmp_path, monkeypatch)
    # A later evaluation crashed after creating its directory. Earlier prefix
    # remains independently scoreable; future garbage is not certified.
    for step in SELECTION_UPDATES:
        if step > 128:
            (tmp_path / 'run/evaluation' / f'selection-{step:04d}/result.json').write_text('incomplete')
    files = {}; result = module.selections(tmp_path, train, policies, files, through_update=128)
    assert [r['updates'] for r in result] == [0, 64, 128] and len(calls) == 3
    assert not any('selection-0192' in p for p in files)


def test_prefix_does_not_skip_an_unfinished_due_selection(tmp_path, monkeypatch):
    train, policies, _ = selection_fixture(tmp_path, monkeypatch)
    path = tmp_path / 'run/evaluation/selection-0064'
    (path / 'result.json').unlink(); path.rmdir()
    with pytest.raises(ValueError, match='selection directories'):
        module.selections(tmp_path, train, policies, {}, through_update=64)


def test_selection_history_across_segments_reuses_original_raw_files_and_rank_bindings(tmp_path,monkeypatch):
    import shutil
    original=tmp_path/'original'
    train,policies,calls=selection_fixture(original,monkeypatch)
    newer=tmp_path/'resumed'
    history=[]
    for step in SELECTION_UPDATES:
        path=original/'run/evaluation'/f'selection-{step:04d}/result.json'
        if step<=128:
            history.append(dict(updates=step,policy_id=policies[step],result=str(path),result_sha256=file_hash(path)))
            continue
        dest=newer/'run/evaluation'/path.parent.name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(path.parent),dest)
        for rank in range(4):
            previous=original/'run'/f'loop-rank-{rank}'/f'selection-{step:04d}.json'
            value=json.loads(previous.read_text());value['result']=str(dest/'result.json')
            output=newer/'run'/f'loop-rank-{rank}'/previous.name;output.parent.mkdir(parents=True,exist_ok=True)
            output.write_text(json.dumps(value))
    files={};reports=module.selections(newer,train,policies,files,history=history)
    assert len(reports)==11 and len(calls)==11
    assert reports[0]['result']==history[0]['result']
    assert str(original/'run/loop-rank-3/selection-0128.json') in files
    assert str(newer/'run/loop-rank-3/selection-0192.json') in files


@pytest.mark.parametrize('damage',[False,True])
def test_resumed_source_compares_saved_master_and_adam_to_audited_prefix(tmp_path,monkeypatch,damage):
    import multimodal_jev.simple_stage2_resume as resume
    config,root,put=stopped_fixture(tmp_path,monkeypatch)
    train=json.loads((root/'config.json').read_text());train['continuous_resume']={'declared':True}
    put(root/'config.json',train)
    launch=json.loads((root/'launch.json').read_text());launch['config_sha256']=file_hash(root/'config.json')
    put(root/'launch.json',launch)
    for path in (root/'config.json',root/'launch.json'):config['pins'][str(path)]=file_hash(path)
    states,_,_=tensor_fixture()
    for rank,state in enumerate(states):
        state.update(stage='simple_stage2_rl',updates=3,next_update=3,schedule_sha256='schedule',
            adapter_id='prefix',reference_policy_id='base:final',algorithm='grpo_v1')
        for adam in state['optimizer']['state'].values():adam['step'].fill_(3)
        saved=copy.deepcopy(state)
        if damage and rank==2:saved['optimizer']['state'][0]['exp_avg']+=1e-6
        out=root/'run'/f'rank-{rank}'
        torch.save(saved,out/'stage2-resumed.pt');(out/'stage2-initial.pt').unlink()
    recovered=dict(completed_updates=3,states=states,files={},initial=dict(reference_policy_id='base:final',files={}))
    monkeypatch.setattr(resume,'authenticate',lambda *a,**k:recovered)
    if damage:
        with pytest.raises(ValueError,match='master/Adam'):module.source(config,verify_parity=None)
    else:
        report=module.source(config,verify_parity=None)
        assert all(c['exact'] for c in report['checks']) and report['recovery'] is recovered


@pytest.mark.parametrize('fault', ['extra', 'missing', 'policy', 'metric', 'hash', 'population'])
def test_selection_orchestration_rejects_unregistered_or_misbound_points(tmp_path, monkeypatch, fault):
    train, policies, _ = selection_fixture(tmp_path, monkeypatch)
    if fault == 'extra': (tmp_path / 'run/evaluation/selection-0001').mkdir()
    elif fault == 'missing':
        path = tmp_path / 'run/evaluation/selection-0000'; (path / 'result.json').unlink(); path.rmdir()
    else:
        path = tmp_path / 'run/loop-rank-3/selection-0634.json'
        value = json.loads(path.read_text())
        if fault == 'policy': value['policy_id'] = 'other'
        elif fault == 'metric': value['metrics']['accuracy'] = 1.
        elif fault == 'hash': value['result_sha256'] = 'changed'
        else: value['recordings'] = 128
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError): module.selections(tmp_path, train, policies, {})

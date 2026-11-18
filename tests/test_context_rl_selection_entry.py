import copy
import importlib.util
import json
from pathlib import Path

import pytest

from tools.jev.run_context_rl_selection import prepare, validate_parity
from tools.jev.context_rl_evaluation_plan import sha


def sources(tmp_path):
    spec = importlib.util.spec_from_file_location('plan_fixture', Path(__file__).with_name('test_context_rl_evaluation_plan.py'))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    args = fixture.fixture(tmp_path)
    contract = dict(base_id='fixture-base', runtime_profile='fixture-runtime',
                    token={'registration_status': 'admitted'}, request_version='jev-context-exclusive_event-v1')
    # Bind all changed fixture artifacts exactly as a collector would. No model
    # execution or runtime admission is claimed by these synthetic fixtures.
    for key in ('stage1', 'rl'):
        root, proof = args[key + '_root'], args[key + '_proof']
        for path in root.glob('run/policy-*/MANIFEST.json'):
            m = json.loads(path.read_text());m['contract'] = contract
            path.write_text(json.dumps(m));proof['files'][str(path.relative_to(root))] = sha(path)
        for path in root.glob('run/checkpoint-*-committed.json'):
            c = json.loads(path.read_text())
            c['joint_policy_manifest_sha256'] = sha(root / 'run' / c['joint_policy_directory'] / 'MANIFEST.json')
            path.write_text(json.dumps(c));proof['files'][str(path.relative_to(root))] = sha(path)
        path = root / 'config.json';c = json.loads(path.read_text())
        if key == 'rl':c['source_contract'] = contract
        path.write_text(json.dumps(c));proof['files']['config.json'] = sha(path)
    template = dict(base_id='fixture-base', context_runtime_profile='fixture-runtime',
        diagnostic_update=False, release_sha256=args['experiment']['release_sha256'],
        source_policy='/old/policy', pins={'/old/policy/weights': 'old', '/runtime/code': 'fixed'},
        audio=[dict(id=str(i)) for i in range(256)])
    return dict(template=template, step=64, **args)


def test_rl_point_uses_real_checkpoint_and_pins(tmp_path):
    args = sources(tmp_path);c = prepare(**args)
    assert c['source_policy_id'] == 'rl-64'
    assert c['source_policy'] == str((args['rl_root'] / 'run/policy-0064').resolve())
    assert '/old/policy/weights' not in c['pins']
    assert c['pins']['/runtime/code'] == 'fixed'
    assert c['rl_evaluation']['source_stage'] == 'task'
    assert c['rl_evaluation']['initial_policy_id'] == 'initial'
    assert args['template']['source_policy'] == '/old/policy'


def test_step_zero_is_stage1_endpoint(tmp_path):
    args = sources(tmp_path);args['step'] = 0
    c = prepare(**args)
    assert c['source_policy_id'] == 'initial'
    assert c['source_policy'].endswith('stage1/run/policy-0634')
    assert c['rl_evaluation']['source_step'] == 634


@pytest.mark.parametrize('key', ['bind_exclusive_event_contract', 'diagnostic_packed_alignment',
    'diagnostic_backbone_alignment', 'capture_hidden', 'parity_microbatch', 'batch_runtime'])
def test_prepare_rejects_diagnostic_overrides(tmp_path, key):
    args = sources(tmp_path);args['template'][key] = 'diagnostic'
    with pytest.raises(ValueError, match='diagnostic execution overrides'):prepare(**args)


@pytest.mark.parametrize('fault', ['mutation', 'step', 'release', 'base', 'runtime', 'update', 'assembled', 'panel'])
def test_prepare_rejects_invalid_sources(tmp_path, fault):
    args = sources(tmp_path);t = args['template']
    if fault == 'mutation':(args['rl_root'] / 'run/policy-0064/execution.pt').write_text('changed')
    elif fault == 'step':args['step'] = 1
    elif fault == 'release':t['release_sha256'] = 'wrong'
    elif fault == 'base':t['base_id'] = 'wrong'
    elif fault == 'runtime':t['context_runtime_profile'] = 'wrong'
    elif fault == 'update':t['diagnostic_update'] = True
    elif fault == 'assembled':t['assembled_execution'] = '/different.pt'
    elif fault == 'panel':t['audio'][1]['id'] = '0'
    with pytest.raises(ValueError):prepare(**args)


def parity():
    c = dict(source_contract={'test': True}, audio=[dict(id=str(i)) for i in range(256)])
    j = dict(policy_id='actual-export', contract=c['source_contract'])
    hf = [dict(id=a['id'], prob=[.25] * 60, inputs={'test': a['id']}, positions=[[1, 2]]) for a in c['audio']]
    vl = [dict(id=x['id'], prob=x['prob'][:], request_identity=dict(
        policy_id=j['policy_id'], inputs=x['inputs'], positions=x['positions'][0])) for x in hf]
    return c, j, copy.deepcopy(hf), copy.deepcopy(vl)


def test_parity_retains_actual_inference_identity():
    args = parity();validate_parity(*args)
    assert args[3][0]['request_identity']['policy_id'] == 'actual-export'


@pytest.mark.parametrize('fault', ['partial', 'input', 'policy', 'positions', 'prob', 'nan', 'range', 'contract'])
def test_parity_rejects_invalid_runtime_records(fault):
    c, j, hf, vl = parity()
    if fault == 'partial':vl.pop()
    elif fault == 'input':vl[0]['request_identity']['inputs'] = {'wrong': True}
    elif fault == 'policy':vl[0]['request_identity']['policy_id'] = 'source-not-export'
    elif fault == 'positions':vl[0]['request_identity']['positions'] = [2, 3]
    elif fault == 'prob':vl[0]['prob'][0] += .01
    elif fault == 'nan':hf[0]['prob'][0] = vl[0]['prob'][0] = float('nan')
    elif fault == 'range':hf[0]['prob'][0] = vl[0]['prob'][0] = 1.1
    elif fault == 'contract':j['contract'] = {'wrong': True}
    with pytest.raises(ValueError):validate_parity(c, j, hf, vl)


@pytest.mark.parametrize('failure', [None, 'hf', 'vllm', 'selection'])
def test_executor_phase_order_and_terminal(tmp_path, monkeypatch, failure):
    import subprocess
    import tools.jev.run_context_rl_selection as entry
    import multimodal_jev.local_runtime_storage as storage
    import multimodal_jev.local_checkpoint as checkpoint
    c, j, hf, vl = parity()
    c.update(source_policy_id='source', checkpoint_step=64, diagnostic_update=False, pins={},
        rl_evaluation=dict(source_policy_id='source', step=64, algorithm='grpo_v1',
            selection_recordings=9226, threshold=.5, binding_files={}))
    cp = tmp_path / 'config.json';cp.write_text(json.dumps(c))
    output = tmp_path / 'output';calls = []
    monkeypatch.setattr(storage, 'require_training_storage', lambda path: None)
    # Isolate storage policy; test real orchestration and filesystem lifetime,
    # not fake model results as evidence for NPU/torch execution.
    monkeypatch.setattr(checkpoint, 'commit_json', lambda p, v: Path(p).write_text(json.dumps(v)))
    monkeypatch.setattr(entry, 'authenticate_export', lambda config, out: j)
    def process(argv, **kwargs):
        phase = argv[argv.index('--phase') + 1] if '--phase' in argv else 'selection'
        calls.append(phase)
        out = Path(argv[argv.index('--output') + 1])
        if phase == 'hf':
            assert not out.exists()  # Real HF CLI uses mkdir(exist_ok=False).
            out.mkdir();(out / 'hf.json').write_text(json.dumps(hf))
        elif phase == 'vllm':
            assert out.exists()
            (out / 'vllm.json').write_text(json.dumps(dict(records=vl)))
        else:
            assert not out.exists()
            config = json.loads(Path(argv[argv.index('--config') + 1]).read_text())
            assert config['policy_id'] == 'actual-export'
            out.mkdir()
        if phase == failure:raise subprocess.CalledProcessError(1, argv)
    monkeypatch.setattr(entry.subprocess, 'run', process)
    if failure:
        with pytest.raises(subprocess.CalledProcessError):entry.execute(cp, output)
        assert not (output / 'point-binding.json').exists()
        assert calls[-1] == failure
    else:
        entry.execute(cp, output)
        assert calls == ['hf', 'vllm', 'selection']
        binding = json.loads((output / 'point-binding.json').read_text())
        assert binding['source_policy_id'] == 'source'
        assert binding['inference_policy_id'] == 'actual-export'
    assert json.loads((output / 'terminal.json').read_text())['exit_code'] == (1 if failure else 0)

"""Lifecycle control tests with fake backend processes, not numerical evidence."""
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('span_rl_step', Path(__file__).parents[1]/'tools/jev/train_span_rl_step.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize('fail_phase', ['before', 'after', None])
def test_lifecycle_gates_update_and_ready(tmp_path, monkeypatch, fail_phase):
    adapter = tmp_path/'initial'
    adapter.mkdir()
    (adapter/'adapter_config.json').write_text('{}')
    calls = []

    def fake_process(argv, check):
        assert check
        if '--phase' in argv:
            phase = argv[argv.index('--phase')+1]
            calls.append(phase)
            if phase == 'update':
                (tmp_path/'update.json').write_text('{}')
                new = tmp_path/'policy-1/adapter'
                new.mkdir(parents=True)
                (new/'adapter_config.json').write_text('{}')
            else:
                for i in range(2):
                    (tmp_path/f'reference-{phase}-{i}.json').write_text('{}')
        else:
            output = Path(argv[argv.index('--output')+1])
            calls.append(output.name)
            phase = output.name.split('-')[1]
            output.mkdir()
            (output/'result.json').write_text(json.dumps({'complete_action_parity':{'passed':phase!=fail_phase}}))
    monkeypatch.setattr(runner.subprocess, 'run', fake_process)
    config = dict(model_path='fixture', initial_adapter=str(adapter), pins={})
    if fail_phase:
        with pytest.raises(ValueError, match='parity failed'):
            runner.orchestrate(config, tmp_path, tmp_path/'config.json')
        assert not (tmp_path/'READY.json').exists()
        assert ('update' in calls) == (fail_phase == 'after')
    else:
        runner.orchestrate(config, tmp_path, tmp_path/'config.json')
        assert (tmp_path/'READY.json').exists()
        assert calls == ['before','rollout-before-0','rollout-before-1','update','after',
                         'rollout-after-0','rollout-after-1']

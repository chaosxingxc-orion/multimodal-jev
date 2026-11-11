"""Lifecycle control tests with fake backend processes, not numerical evidence."""
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('span_rl_step', Path(__file__).parents[1]/'tools/jev/train_span_rl_step.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_resume_cursor_rejects_repeat_skip_and_changed_data():
    state=dict(schema='jev-span-balanced-cursor-v1',seed=17,schedule_sha256='schedule',
               epoch_batches=20,manifest_sha256='data',question_order=['a','b','c'],
               batch_index=0,next_batch_index=1)
    current={**state,'batch_index':1,'next_batch_index':2}
    runner.validate_resume_cursor({'training_state':state},current,{})
    for index in (0,2):
        with pytest.raises(ValueError,match='next balanced pair'):
            runner.validate_resume_cursor({'training_state':state},{**current,'batch_index':index},{})
    with pytest.raises(ValueError,match='schedule changed'):
        runner.validate_resume_cursor({'training_state':state},{**current,'manifest_sha256':'other'}, {})
    with pytest.raises(ValueError,match='pinned selection evidence'):
        runner.validate_resume_cursor({},current,{})


def test_legacy_cursor_requires_matching_checkpoint_and_file_pin(tmp_path):
    state=dict(schema='jev-span-balanced-cursor-v1',seed=17,schedule_sha256='schedule',
               epoch_batches=20,manifest_sha256='data',question_order=['a','b','c'],
               batch_index=0,next_batch_index=1)
    current={**state,'batch_index':1,'next_batch_index':2}
    path=tmp_path/'legacy.json'
    path.write_text(json.dumps(dict(schema='jev-span-legacy-selection-v1',
        checkpoint_optimizer_sha256='optimizer',training_state=state)))
    config=dict(legacy_selection_evidence=str(path),pins={str(path):runner.file_hash(path)})
    runner.validate_resume_cursor({'optimizer_sha256':'optimizer'},current,config)
    with pytest.raises(ValueError,match='checkpoint mismatch'):
        runner.validate_resume_cursor({'optimizer_sha256':'other'},current,config)
    path.write_text('{}')
    with pytest.raises(ValueError,match='pinned selection evidence'):
        runner.validate_resume_cursor({'optimizer_sha256':'optimizer'},current,config)


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

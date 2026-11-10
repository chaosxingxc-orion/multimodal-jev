import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from multimodal_jev.baseline import file_hash

spec = importlib.util.spec_from_file_location('selected_supervisor',
    Path(__file__).parents[1] / 'tools/jev/run_selected_frozen_test.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def setup(tmp_path):
    root = tmp_path / 'evaluation'; root.mkdir()
    baseline = tmp_path / 'baseline'; baseline.mkdir()
    (baseline / 'launch.json').write_text(json.dumps({'container_id': 'baseline'}))
    (root / 'instruction.txt').write_text('frozen instruction')
    (root / 'selection.json').write_text(json.dumps({'selected_instruction_sha256': file_hash(root / 'instruction.txt')}))
    (root / 'config-template.json').write_text('{"split":"train"}')
    plan = {'baseline_root': str(baseline), 'baseline_container_id': 'baseline', 'image': 'image',
            'selection_sha256': file_hash(root / 'selection.json'),
            'config_template_sha256': file_hash(root / 'config-template.json'),
            'deadline_unix': 9999999999, 'prior_device_container_id': 'previous',
            'argv': ['docker', 'run'], 'source_commit': 'commit', 'physical_device': 7,
            'targets': 'train.targets.jsonl', 'six_views': True}
    (root / 'plan.json').write_text(json.dumps(plan))
    return root


def test_changed_configuration_is_rejected_before_wait_or_dispatch(tmp_path, monkeypatch):
    root = setup(tmp_path)
    (root / 'config-template.json').write_text('{"split":"test"}')
    monkeypatch.setattr(module, 'wait_container', lambda *args: pytest.fail('must fail before waiting'))
    with pytest.raises(ValueError, match='configuration'): module.execute(root)
    assert not (root / 'dispatch-intent.json').exists()


def test_six_view_scores_wait_for_both_containers_and_dispatch_once(tmp_path, monkeypatch):
    root = setup(tmp_path); events = []
    def wait(cid, *args):
        events.append(('completed', cid))
        return {'State': {'Status': 'exited', 'ExitCode': 0}}
    def export(run, path):
        assert events == [('completed', 'baseline')]
        path.write_text('{}')
    def run(argv, **kwargs):
        if argv[0] == 'ssh':
            assert (root / 'dispatch-intent.json').exists()
            events.append(('dispatch', None))
            return SimpleNamespace(returncode=0, stdout='a' * 64, stderr='')
        assert events[-1] == ('completed', 'a' * 64)
        assert '--six-views' in argv
        Path(argv[argv.index('--output') + 1]).write_text('{}')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module, 'wait_container', wait)
    monkeypatch.setattr(module, 'export_evidence', export)
    monkeypatch.setattr(module, 'observe', lambda *args: {'State': {'Status': 'exited'}})
    monkeypatch.setattr(module.subprocess, 'run', run)
    module.execute(root)
    assert (root / 'baseline-score.json').exists() and (root / 'selected-score.json').exists()
    assert json.loads((root / 'terminal.json').read_text())['exit_code'] == 0
    events.clear()
    with pytest.raises(ValueError, match='previous dispatch'): module.execute(root)
    assert ('dispatch', None) not in events

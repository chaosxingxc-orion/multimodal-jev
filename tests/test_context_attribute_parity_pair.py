import hashlib
import json
import subprocess

import pytest

from tools.reports import run_context_attribute_parity_pair as pair
from tests.test_context_attribute_parity import reports


def package(tmp_path):
    root = tmp_path/'package'
    root.mkdir()
    config = root/'probe-config.json'
    config.write_text('{}')
    (root/'FILES.json').write_text(json.dumps({config.name:hashlib.sha256(config.read_bytes()).hexdigest()}))
    return root


def test_package_rejects_changed_and_extra_payload(tmp_path):
    root = package(tmp_path)
    pair.verify(root)
    extra = root/'unexpected'
    extra.write_text('extra')
    with pytest.raises(ValueError):pair.verify(root)
    extra.unlink()
    (root/'probe-config.json').write_text('{"changed":true}')
    with pytest.raises(ValueError):pair.verify(root)


@pytest.mark.parametrize('failure', [None, 'hf', 'vllm'])
def test_fresh_process_order_and_failure_terminal(tmp_path, monkeypatch, failure):
    import multimodal_jev.local_runtime_storage as storage
    import multimodal_jev.local_checkpoint as checkpoint
    from tools.reports.probe_context_attribute_parity import compare
    root = package(tmp_path)
    out = tmp_path/'evaluation'
    monkeypatch.setattr(storage, 'require_training_storage', lambda path:None)
    monkeypatch.setattr(checkpoint, 'commit_json', lambda path,value:path.write_text(json.dumps(value)))
    calls = []
    hf, vl = reports()
    def invoke(argv, **kwargs):
        phase = argv[-1]
        calls.append(phase)
        assert kwargs['cwd'] == root and kwargs['check'] is True
        assert argv[1:3] == ['-m', 'tools.reports.probe_context_attribute_parity']
        if phase == failure:raise subprocess.CalledProcessError(1, argv)
        (out/(phase+'.json')).write_text(json.dumps(hf if phase == 'hf' else vl))
        if phase == 'vllm':(out/'parity.json').write_text(json.dumps(compare(hf, vl)))
    monkeypatch.setattr(pair.subprocess, 'run', invoke)
    if failure:
        with pytest.raises(subprocess.CalledProcessError):pair.run(root, out)
    else:
        assert pair.run(root, out)['exact']
    terminal = json.loads((out/'terminal.json').read_text())
    assert terminal['exit_code'] == int(failure is not None)
    assert calls == (['hf'] if failure == 'hf' else ['hf', 'vllm'])
    if failure:assert terminal['phase'] == failure
    with pytest.raises(FileExistsError):pair.run(root, out)

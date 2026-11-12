import os
from contextlib import nullcontext
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/jev'))
import run_intent_binding_matrix as module
from multimodal_jev.baseline import atomic_json, file_hash


def predecessor(tmp_path):
    root = tmp_path/'prior'; root.mkdir(); (root/'run').mkdir()
    atomic_json(root/'config.json', {})
    atomic_json(root/'launch.json', dict(pid=os.getpid(),
        argv=Path(f'/proc/{os.getpid()}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')))
    return dict(predecessor_controller=str(root),
        pins={str(root/n):file_hash(root/n) for n in ('launch.json','config.json')})


def test_live_predecessor_waits_ten_minutes_then_failure_stops(tmp_path, monkeypatch):
    config = predecessor(tmp_path); waits = []
    def sleep(seconds):
        waits.append(seconds)
        atomic_json(Path(config['predecessor_controller'])/'run/terminal.json', dict(exit_code=1))
    monkeypatch.setattr(module.time, 'sleep', sleep)
    with pytest.raises(ValueError, match='predecessor failed'):
        module.await_predecessor(config, tmp_path)
    assert waits == [600]


def test_missing_predecessor_is_not_completion(tmp_path):
    config = predecessor(tmp_path); path = Path(config['predecessor_controller'])/'launch.json'
    atomic_json(path, dict(pid=999999999, argv=['missing']))
    config['pins'][str(path)] = file_hash(path)
    with pytest.raises(ValueError, match='missing without terminal'):
        module.await_predecessor(config, tmp_path)


@pytest.mark.parametrize('fail_at', [None, 'step0', 'step4'])
def test_every_gate_must_be_accepted_before_scoring(tmp_path, monkeypatch, fail_at):
    root = tmp_path/'matrix'; root.mkdir(); out = tmp_path/'out'; out.mkdir()
    entries = [dict(key=k, root=str(root/k)) for k in ('step0','step4')]
    path = root/'protocol.json'
    atomic_json(path, dict(image='fixed', checkpoints=entries)); digest = file_hash(path)
    pins = {str(path):digest}
    for entry in entries:
        gate = Path(entry['root']); gate.mkdir()
        atomic_json(gate/'config.json', dict(pins={str(path):digest}))
        atomic_json(gate/'source-manifest.json', {})
        atomic_json(gate/'prepared.json', dict(root=str(gate), image='fixed', physical_devices=[4],
            create_argv=[], config_sha256=file_hash(gate/'config.json'),
            source_manifest_sha256=file_hash(gate/'source-manifest.json')))
        pins.update({str(gate/n):file_hash(gate/n) for n in ('config.json','prepared.json','source-manifest.json')})
    events = []
    monkeypatch.setattr(module, 'wait_resources', lambda *a: None)
    monkeypatch.setattr(module, 'locks', lambda *a: nullcontext())
    def launch(gate, *args):
        events.append(('launch', gate.name)); return dict(container_id=gate.name)
    def wait(cid, *args): events.append(('exit', cid))
    def accept(entry, *args):
        events.append(('accept', entry['key']))
        if entry['key'] == fail_at: raise ValueError('backend failure')
        return {}, dict(passed=True)
    def score(*args):
        assert events == [(phase,k) for k in ('step0','step4') for phase in ('launch','exit','accept')]
        events.append(('score','all')); return dict(done=True)
    monkeypatch.setattr(module, 'launch', launch); monkeypatch.setattr(module, 'wait_exact', wait)
    monkeypatch.setattr(module, 'authenticate', accept); monkeypatch.setattr(module, 'score', score)
    config = dict(protocol=str(path), protocol_sha256=digest, image='fixed', pins=pins, lock_directory=str(tmp_path))
    if fail_at:
        with pytest.raises(ValueError, match='backend failure'): module.run_matrix(config,out)
        assert events[-1] == ('accept', fail_at) and not (out/'comparison.json').exists()
    else:
        module.run_matrix(config,out)
        assert module.read(out/'comparison.json')['done']
    # Even a prior uncertain create must prevent any second attempt.
    atomic_json(root/'step0/create-attempt.json', {})
    before = list(events)
    with pytest.raises(ValueError, match='already claimed'): module.run_matrix(config,out)
    assert events == before

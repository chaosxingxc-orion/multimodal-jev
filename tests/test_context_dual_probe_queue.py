import json

import pytest

from tools.reports import queue_context_dual_probe as queue


def setup(tmp_path, monkeypatch):
    c = dict(output_directory=str(tmp_path/'queue'), predecessor_process={})
    calls = []
    monkeypatch.setattr(queue, 'validate', lambda c: ({}, {}))
    monkeypatch.setattr(queue, 'check_paths', lambda c: None)
    monkeypatch.setattr(queue, 'wait_exact', lambda p: calls.append('wait'))
    monkeypatch.setattr(queue.probe, 'check_priority', lambda p: calls.append('priority'))
    def launch(*args):
        calls.append('launch')
        return dict(status='finite_fixed_policy_inference_audited',
                    exact_actual_logits=True, exact_probabilities_and_G16=True)
    monkeypatch.setattr(queue.probe, 'launch', launch)
    return c, calls


def test_wait_and_success_validation_precede_single_launch(tmp_path, monkeypatch):
    c, calls = setup(tmp_path, monkeypatch)
    queue.run(c, {})
    assert calls == ['wait', 'priority', 'launch']
    status = json.loads((tmp_path/'queue/status.json').read_text())
    assert status['phase'] == 'complete_finite_audio_probe'
    assert not status['training_dispatch_allowed']
    with pytest.raises(FileExistsError):
        queue.run(c, {})
    assert calls == ['wait', 'priority', 'launch']


@pytest.mark.parametrize('stage', ['wait', 'priority', 'launch', 'audit'])
def test_failure_preserves_without_retry(tmp_path, monkeypatch, stage):
    c, calls = setup(tmp_path, monkeypatch)
    def fail(*args):
        raise ValueError('expected refusal')
    if stage == 'wait': monkeypatch.setattr(queue, 'wait_exact', fail)
    elif stage == 'priority': monkeypatch.setattr(queue.probe, 'check_priority', fail)
    elif stage == 'launch': monkeypatch.setattr(queue.probe, 'launch', fail)
    else: monkeypatch.setattr(queue.probe, 'launch', lambda *args: {})
    with pytest.raises(ValueError): queue.run(c, {})
    status = json.loads((tmp_path/'queue/status.json').read_text())
    assert status['phase'] == 'stopped_preserve_handles'
    assert status['restart'] is False
    assert 'launch' not in calls


@pytest.mark.parametrize('bad', ['unpinned', 'idle', 'disabled', 'other_root'])
def test_validation_rejects_unsafe_binding(monkeypatch, bad):
    c = dict(schema='context-dual-probe-queue-v1', pins={'config':'x', 'local':'y'},
             probe_config='config', probe_local='local', predecessor_root='/research')
    p = dict(priority_mode='completed_research_chain',
             finite_inference_dispatch_allowed=True, priority_root='/research')
    if bad == 'unpinned': c['pins'].pop('local')
    if bad == 'idle': p['priority_mode'] = 'idle_resource_window'
    if bad == 'disabled': p['finite_inference_dispatch_allowed'] = False
    if bad == 'other_root': p['priority_root'] = '/other'
    monkeypatch.setattr(queue, 'verify', lambda p: None)
    monkeypatch.setattr(queue, 'load_config', lambda *args: (p, {}))
    with pytest.raises(ValueError): queue.validate(c)

import copy
import json
from pathlib import Path
import sys
import subprocess
import pytest
from multimodal_jev.resource_observers import is_audited_completion_observer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/jev'))
import queue_span_dp4_resume as queue


def observer(kind='observer'):
    record = json.loads((Path(__file__).resolve().parents[1]/f'docs/engineering/intent-resource-{kind}-inspect-v1.json').read_text())
    record['HostConfig'] = record.pop('host_config')
    return record


def test_exact_listener_is_identified():
    assert is_audited_completion_observer(observer())
    assert is_audited_completion_observer(observer('archive'))


@pytest.mark.parametrize('kind', ['observer', 'archive'])
@pytest.mark.parametrize('bad', ['program', 'argument', 'path', 'image', 'device', 'bind', 'extra', 'pid', 'remove'])
def test_unknown_privileged_program_is_not_exempt(bad,kind):
    row = observer(kind)
    if bad == 'program': row['Args'][7] += '\nimport torch_npu\n'
    if bad == 'argument': row['Args'][5] = '/different/python'
    if bad == 'path': row['Args'][8] = '/var/tmp/omni-local-state/../different'
    if bad == 'image': row['Image'] = 'unknown'
    if bad == 'device': row['HostConfig']['Devices'] = [{'PathOnHost': '/dev/davinci4'}]
    if bad == 'bind': row['HostConfig']['Binds'].append('/extra:/extra')
    if bad == 'extra': row['Args'].append('extra')
    if bad == 'pid': row['HostConfig']['PidMode'] = 'private'
    if bad == 'remove': row['HostConfig']['AutoRemove'] = False
    assert not is_audited_completion_observer(row)


@pytest.mark.parametrize('busy', [False, True])
@pytest.mark.parametrize('kind', ['observer', 'archive'])
def test_listener_never_bypasses_actual_device_fd_check(monkeypatch, busy,kind):
    row = observer(kind); calls = []
    monkeypatch.setattr(queue, 'inspect', lambda ids: {row['Id']: row})
    def remote(argv):
        calls.append(argv)
        if argv[:2] == ['docker', 'ps']:
            return subprocess.CompletedProcess(argv, 0, row['Id'], '')
        assert argv[-1] == queue.FD_CHECK
        return subprocess.CompletedProcess(argv, int(busy), json.dumps({'busy': [[123, '4', 4]] if busy else [], 'denied': []}), '')
    monkeypatch.setattr(queue, 'remote', remote)
    if busy:
        with pytest.raises(subprocess.CalledProcessError): queue.check_devices('fixed')
    else: queue.check_devices('fixed')
    assert len(calls) == 2


def test_archive_requires_exact_second_path():
    row=observer('archive');row['Args'][9]+='/different'
    assert not is_audited_completion_observer(row)


def test_same_completion_listener_at_audited_runtime_only():
    row=observer();row['Args'][8]='/var/tmp/s1-bl-candidate-ops-media-v1/runtime'
    assert is_audited_completion_observer(row)
    row['Args'][8]+='/other'
    assert not is_audited_completion_observer(row)

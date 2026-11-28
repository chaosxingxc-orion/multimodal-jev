"""Recover an existing container's terminal evidence without launching a job."""
import json
import shlex
import subprocess

import pytest

from multimodal_jev.local_checkpoint import commit_json
from tools.reports import native_seal_operations as operations
from tools.reports.context_macro_probe_operations import IMAGE


@pytest.mark.parametrize('exit_code', [0, 137])
def test_reattach_collects_exact_exit_without_dispatch(tmp_path, monkeypatch, exit_code):
    monkeypatch.setattr('multimodal_jev.local_checkpoint.check_paths', lambda _: None)
    cid = 'a'*64
    image = IMAGE
    name = 'example-formal'
    remote_root = '/local/example'
    argv = ['docker', 'create', '--name', name, '--entrypoint', '/entrypoint',
            '--memory', str(2**40), '-v', remote_root+':/jev-work', '-e', 'EXAMPLE=1',
            '--device', '/dev/davinci0:/dev/davinci0', image, 'python']
    handle = dict(container_id=cid, argv=argv, host='test', remote_root=remote_root)
    pins = {'formal.json': 'example'}
    for file, value in (
        ('formal-launch.json', handle), ('formal-intent.json', {'argv': argv}),
        ('formal-admission.json', {'entry_pins': pins}), ('finite-wait.json', {'exit_code': 0}),
        ('finite-collected.json', {'passed': True}),
    ):
        commit_json(tmp_path/file, value)
    lease = tmp_path/'lease'
    lease.touch()
    c = dict(chains=[[[dict(name='finite', collect='finite/receipt.json')],
                      [dict(name='formal', devices=[0], collect='formal/receipt.json')]]],
             device_group_lease=True, operations=str(tmp_path), lease=str(lease), tag='example',
             host=handle['host'], remote_root=remote_root, image=image, entry_pins=pins)
    waited = False
    receipt = {'passed': True, 'updates': 150}

    def ssh(command, **_):
        nonlocal waited
        args = shlex.split(command[-1])
        if args == ['docker', 'inspect', cid]:
            state = dict(Status='exited' if waited else 'running', Running=not waited,
                         Pid=0 if waited else 123, ExitCode=exit_code if waited else 0, OOMKilled=False)
            row = dict(Id=cid, Name='/'+name, Image=image, State=state, RestartCount=0,
                       Config=dict(Entrypoint=['/entrypoint'], Cmd=['python'], Env=['EXAMPLE=1']),
                       HostConfig=dict(Privileged=False, NetworkMode='none',
                           Binds=[remote_root+':/jev-work'], Memory=2**40,
                           Devices=[dict(PathOnHost='/dev/davinci0', PathInContainer='/dev/davinci0',
                                         CgroupPermissions='rwm')]))
            output = json.dumps([row])
        elif args == ['docker', 'wait', cid]:
            waited = True
            output = str(exit_code)
        elif args == ['docker', 'logs', cid]:
            output = 'original logs'
        else:
            assert exit_code == 0 and args == ['cat', remote_root+'/formal/receipt.json']
            output = json.dumps(receipt)
        return subprocess.CompletedProcess(command, 0, output, '')

    monkeypatch.setattr(operations.subprocess, 'run', ssh)
    assert operations.collect_existing_final(c) == exit_code
    terminal = json.loads((tmp_path/'terminal.json').read_bytes())
    assert terminal['exit_code'] == exit_code
    if exit_code == 0:
        assert terminal['status'] == 'declared_stages_finished'
        assert json.loads((tmp_path/'formal-collected.json').read_bytes()) == receipt
    else:
        assert not (tmp_path/'formal-collected.json').exists()
        assert terminal['restart'] is False

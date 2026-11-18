import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from tools.jev import collect_context_rl_training as module


@pytest.fixture
def transport(tmp_path, monkeypatch):
    remote=tmp_path/'remote';remote.mkdir()
    (remote/'source').mkdir();(remote/'source/code.py').write_text('# fixed')
    pins={str(remote/'source/code.py'):module.sha(remote/'source/code.py')}
    (remote/'config.json').write_text(json.dumps(dict(pins=pins)))
    (remote/'service-config.json').write_text('{}');(remote/'run.py').write_text('# runner')
    manifest=dict(files={str(p.relative_to(remote)):module.sha(p) for p in remote.rglob('*') if p.is_file()})
    (remote/'PREPARED.json').write_text(json.dumps(manifest))
    for directory in ('run','service','supervision','reference-cache'):
        (remote/directory).mkdir();(remote/directory/'result.json').write_text(json.dumps({'directory':directory}))
    (remote/'run/rank-0').mkdir();(remote/'run/rank-0/checkpoint.pt').write_bytes(b'checkpoint')
    launch=dict(container_id='exact-handle',image='fixed-image',remote_root=str(remote),
        config_sha256=module.sha(remote/'config.json'),service_config_sha256=module.sha(remote/'service-config.json'),
        manifest_sha256=module.sha(remote/'PREPARED.json'))
    observed=dict(Id='exact-handle',Image='fixed-image',State=dict(Status='exited',ExitCode=0,OOMKilled=False))
    calls=[];real_run=subprocess.run
    def decode(args):
        assert args[:2]==['ssh','npu'];command=shlex.split(args[2]);calls.append(command)
        return command
    def output(args,**kwargs):
        command=decode(args)
        if command[:2]==['docker','inspect']:return json.dumps([observed])
        assert '--volumes-from' in command and 'exact-handle:ro' in command
        assert '--device' not in command and '--network=none' in command
        return real_run([sys.executable,'-c',command[-1]],text=True,stdout=subprocess.PIPE,check=True).stdout
    def run(args,**kwargs):
        command=decode(args)
        assert 'exact-handle:ro' in command
        return real_run([sys.executable,'-c',command[-1]],**kwargs)
    monkeypatch.setattr(module.subprocess,'check_output',output)
    monkeypatch.setattr(module.subprocess,'run',run)
    return remote,launch,observed,tmp_path/'collected',calls


@pytest.mark.parametrize('exit_code',[0,1])
def test_real_reader_and_archive_transport_preserve_terminal_outputs(transport,exit_code):
    remote,launch,observed,out,calls=transport;observed['State']['ExitCode']=exit_code
    result=module.collect(launch,out)
    assert result['state']['ExitCode']==exit_code
    assert result['status']=='transport_complete_independent_audit_required'
    assert not result['bad_pins'] and not result['bad_package']
    assert (out/'run/rank-0/checkpoint.pt').read_bytes()==b'checkpoint'
    assert all(module.sha(out/p)==digest for p,digest in result['files'].items())
    assert not (out/'source').exists() and out.with_suffix('.tar').exists()


@pytest.mark.parametrize('fault',['live','image','handle'])
def test_wrong_or_live_container_not_collected(transport,fault):
    _,launch,observed,out,calls=transport
    if fault=='live':observed['State']['Status']='running'
    elif fault=='image':observed['Image']='different'
    else:observed['Id']='different'
    with pytest.raises(ValueError):module.collect(launch,out)
    assert len(calls)==1 and not out.exists()


@pytest.mark.parametrize('fault',['source_pin','config_identity','service_identity'])
def test_drift_preserved_before_rejection(transport,fault):
    remote,launch,_,out,_=transport
    if fault=='source_pin':(remote/'source/code.py').write_text('# changed')
    elif fault=='config_identity':launch['config_sha256']='wrong'
    else:launch['service_config_sha256']='wrong'
    with pytest.raises(ValueError,match='collection preserved'):module.collect(launch,out)
    receipt=json.loads(out.with_suffix('.transport.json').read_text())
    assert receipt['status']=='transport_complete_independent_audit_required'
    assert (out/'run/rank-0/checkpoint.pt').exists()
    assert receipt['bad_pins'] or receipt['launch_binding_mismatches']


def test_remote_symlink_rejected(transport):
    remote,launch,_,out,_=transport
    (remote/'run/link').symlink_to(remote/'config.json')
    with pytest.raises(subprocess.CalledProcessError):module.collect(launch,out)
    assert not out.exists()

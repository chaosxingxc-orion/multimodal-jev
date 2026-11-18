"""Exercise the real remote archive program using a local subprocess substitute."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
from tools.reports import audit_context_macro_probe as m


def fixture(tmp_path,monkeypatch):
    remote=tmp_path/'remote';ops=tmp_path/'ops';ops.mkdir()
    for name in ('run','journal','service-0','service-1'):
        (remote/name).mkdir(parents=True)
        (remote/name/'result').write_bytes((name+' payload').encode())
    (ops/'terminal.json').write_text('{"exit_code":0}')
    for role in ('preflight','client','service-0','service-1'):
        (ops/(role+'-launch.json')).write_text(json.dumps(dict(container_id=role,argv=['create',role])))
    monkeypatch.setattr(m,'check_paths',lambda *a:[])
    monkeypatch.setattr(m,'inspect_runtime',lambda *a,**kw:None)
    calls=[];real=subprocess.run
    def run(argv,**kw):
        calls.append((argv,kw.get('timeout')))
        if argv[0]=='ssh':
            args=shlex.split(argv[-1])
            if args[:2]==['docker','inspect']:
                return subprocess.CompletedProcess(argv,0,json.dumps([{'Id':args[2]}]),'')
            if 'actual=verify(' in args[-1]:
                return subprocess.CompletedProcess(argv,0,'sealed\n','')
            return real([sys.executable,'-B','-c',args[-1]],**kw)
        if argv[0]=='scp':
            Path(argv[-1]).write_bytes((remote/'results.tar').read_bytes())
            return subprocess.CompletedProcess(argv,0,b'',b'')
        raise AssertionError(argv)
    monkeypatch.setattr(m.subprocess,'run',run)
    c=dict(operations_root=str(ops),remote_root=str(remote),host='fixture',prepared=str(tmp_path),collection_transfer_timeout_seconds=123)
    return c,remote,ops,calls


def collect(c):
    return m.collect_and_audit(c,validator=lambda c:dict(files_sha256='sealed',image='image'),
        command_builder=lambda c,role:['create',role],result_dirs=('run','journal','service-0','service-1'),
        output_auditor=lambda p,r:dict(status='fixture_audited'))


def test_complete_streamed_archive_is_hashed_and_collected(tmp_path,monkeypatch):
    c,remote,ops,calls=fixture(tmp_path,monkeypatch);result=collect(c)
    assert result['status']=='fixture_audited'
    wire=json.loads((ops/'transport.json').read_text())
    assert len(wire['files'])==4
    for name,digest in wire['files'].items():
        assert hashlib.sha256((ops/'collected'/name).read_bytes()).hexdigest()==digest
    assert wire['archive_sha256']==m.file_hash(ops/'results.tar')
    assert any(argv[0]=='scp' and timeout==123 for argv,timeout in calls)
    assert any(argv[0]=='ssh' and timeout==123 for argv,timeout in calls)


def test_remote_enumeration_error_retains_diagnostic_and_creates_no_archive(tmp_path,monkeypatch):
    c,remote,ops,_=fixture(tmp_path,monkeypatch)
    (remote/'run/link').symlink_to(remote/'journal/result')
    with pytest.raises(RuntimeError,match='symlink in result'):collect(c)
    assert not (remote/'results.tar').exists()
    assert not (ops/'audit.json').exists()


@pytest.mark.parametrize('timeout',[True,0,54,7201,'123'])
def test_invalid_timeout_refused_before_remote_work(tmp_path,monkeypatch,timeout):
    c,remote,ops,calls=fixture(tmp_path,monkeypatch);c['collection_transfer_timeout_seconds']=timeout
    with pytest.raises(ValueError,match='timeout'):collect(c)
    assert not calls

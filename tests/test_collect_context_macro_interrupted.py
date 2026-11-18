"""Actual archive/transport program with SSH substituted; prefix/model fixtures."""
import json
import pytest

from test_context_macro_collection_transport import fixture
from tools.reports import collect_context_macro_interrupted as m
from tools.reports import prepare_context_macro_sparse_resume as prepare


def setup(tmp_path,monkeypatch):
    c,remote,ops,calls=fixture(tmp_path,monkeypatch)
    (ops/'terminal.json').write_text('{"exit_code":1,"error":"actor failed"}')
    c['recovery_step']=64
    monkeypatch.setattr(m,'validate',lambda c:dict(files_sha256='sealed',image='image'))
    monkeypatch.setattr(m,'command',lambda c,role:['create',role])
    # Separate tests below exercise actual exited-state rejection.
    monkeypatch.setattr(m,'inspect_exited',lambda *a,**kw:None)
    monkeypatch.setattr(m,'audit_prefix',lambda *a:dict(status='formal_dp2_prefix_outputs_audited'))
    monkeypatch.setattr(prepare,'checkpoint',lambda *a:({}, {}, set(), {'prefix_learning_history_verified':True}))
    return c,remote,ops,calls


def test_failed_formal_attempt_preserves_failure_and_collects_bound_prefix(tmp_path,monkeypatch):
    c,remote,ops,calls=setup(tmp_path,monkeypatch)
    result=m.collect(c)
    assert result['parent_training_complete'] is False
    assert json.loads((ops/'terminal.json').read_text())['exit_code']==1
    assert len(json.loads((ops/'transport.json').read_text())['files'])==4
    assert (ops/'prefix-audit.json').exists() and (ops/'recovery-prefix-evidence.json').exists()
    assert json.loads((ops/'prefix-audit.json').read_text())['parent_training_complete'] is False
    assert not (ops/'audit.json').exists()
    assert not any('docker stop' in str(argv) or 'docker start' in str(argv) for argv,_ in calls)


def test_bad_prefix_preserves_collected_failure_without_recovery_evidence(tmp_path,monkeypatch):
    c,remote,ops,calls=setup(tmp_path,monkeypatch)
    def fail(*a):raise ValueError('bad prefix')
    monkeypatch.setattr(m,'audit_prefix',fail)
    with pytest.raises(ValueError,match='bad prefix'):m.collect(c)
    assert (ops/'transport.json').exists() and (ops/'collected-inspect.json').exists()
    assert not (ops/'recovery-prefix-evidence.json').exists()


@pytest.mark.parametrize('state,role,allowed',[
    (dict(Status='exited',Running=False,ExitCode=1,OOMKilled=True),'client',True),
    (dict(Status='running',Running=True,ExitCode=0,OOMKilled=False),'client',False),
    (dict(Status='paused',Running=False,ExitCode=0,OOMKilled=False),'client',False),
    (dict(Status='exited',Running=False,ExitCode=1,OOMKilled=False),'preflight',False),
    (dict(Status='exited',Running=False,ExitCode=0,OOMKilled=False),'preflight',True),
])
def test_exact_terminal_state_required(monkeypatch,state,role,allowed):
    seen=[]
    monkeypatch.setattr(m,'inspect_runtime',lambda *a:seen.append(a))
    if allowed:m.inspect_exited({'State':state},{'role':role},terminal=True)
    else:
        with pytest.raises(ValueError):m.inspect_exited({'State':state},{'role':role},terminal=True)
    assert len(seen)==1


def test_invalid_sparse_cursor_never_calls_remote(tmp_path,monkeypatch):
    c,remote,ops,calls=setup(tmp_path,monkeypatch);c['recovery_step']=317
    with pytest.raises(ValueError):m.collect(c)
    assert calls==[]

import json,os,socket,struct,subprocess,sys,threading
from pathlib import Path
import pytest
from tools.reports.context_runtime_startup import close_replica,supervise_services
from tools.reports.prepare_context_replica_training_candidate import RUN_PY,ENGINEERING_GATE
from test_context_process_supervisor import SERVICE


GREETING=SERVICE.replace('command=json.loads', '''hello=dict(protocol='replica-macro-runtime-hello-v1',max_request_queries=192,
 runtime_identity={'hash':'h'},policy_id='a'*64,contract={'large':'x'*10000})
b=json.dumps(hello).encode();c.sendall(struct.pack('!I',len(b))+b)
command=json.loads''')


def test_actual_replica_greeting_before_close_after_actor_failure(tmp_path):
    address=tmp_path/'s';ready=tmp_path/'ready'
    report=supervise_services(services=[dict(name='a',command=[sys.executable,'-c',GREETING,str(address),str(ready)],
        env=dict(os.environ),ready_path=ready,socket_path=address)],actor_command=[sys.executable,'-c','raise SystemExit(7)'],
        actor_env=dict(os.environ),output=tmp_path/'supervision',startup_timeout=3.,shutdown_timeout=2.)
    assert report['actor_exit_code']==7 and not report['passed']
    assert report['services'][0]['exit_code']==0 and report['services'][0]['close_ack']['ok']
    assert not report['forced_cleanup']


def test_runner_preflight_precedes_all_service_launches():
    from tools.reports import context_stage2_runtime_runner
    source=Path(context_stage2_runtime_runner.__file__).read_text()
    assert source.index("command_runner(launch['preflight_command']")<source.index('services=[]')<source.index('return supervisor(')
    assert 'check=True' in source


def test_recovery_runner_also_uses_safe_environment(monkeypatch):
    from tools.reports.context_audit_release import acceptance_graph
    with pytest.raises(RuntimeError,match='all-stage'):
        acceptance_graph(None)

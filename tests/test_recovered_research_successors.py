import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.reports import continue_recovered_research as successor


def fixture(tmp_path,monkeypatch):
    root=tmp_path/'predecessor';root.mkdir();(root/'owner.lock').touch()
    (root/'status.json').write_text(json.dumps(dict(phase='complete_training_and_selection_curve')))
    lock=tmp_path/'successor.lock';lock.touch()
    stopped=tmp_path/'old-status.json';stopped.write_text(json.dumps(dict(phase='stopped_preserve_handles')))
    pin=tmp_path/'source.py';pin.write_text('unchanged')
    target=tmp_path/'persistent'/'rloo';alias=tmp_path/'rloo'
    result=target/'status.json'
    c=dict(schema='recovered-research-successors-v1',output_directory=str(tmp_path/'recovery'),
        predecessor_process={},predecessor_root=str(root),pins={str(pin):successor.sha(pin)},
        stopped_states={str(stopped):successor.sha(stopped)},owner_locks=[str(lock)],
        absent_paths=[str(alias)],aliases={str(alias):str(target)},stages=[dict(name='rloo',
          module='tools.reports.run_context_arithmetic_experiment',arguments=['--spec','original','--output',str(alias),'--launch'],
          storage_paths=[str(target)],results=[dict(path=str(result),equals=dict(phase='complete'))])])
    monkeypatch.setattr(successor,'load_config',lambda *args:(c,{}))
    monkeypatch.setattr(successor,'check_paths',lambda *args:None)
    monkeypatch.setattr(successor,'wait_exact',lambda process:None)
    return c,target,result


def test_successor_keeps_original_specs_and_failures(tmp_path,monkeypatch):
    c,target,result=fixture(tmp_path,monkeypatch);calls=[]
    def invoke(argv,**kw):
        calls.append(argv);target.mkdir();result.write_text(json.dumps(dict(phase='complete')))
    monkeypatch.setattr(successor.subprocess,'run',invoke)
    successor.run('template','local')
    assert calls[0][2]=='tools.reports.context_compatible_pipeline'
    assert calls[0][-5:]==['--spec','original','--output',str(tmp_path/'rloo'),'--launch']
    assert (tmp_path/'rloo').resolve()==target
    assert json.loads((tmp_path/'old-status.json').read_text())['phase']=='stopped_preserve_handles'
    assert json.loads((tmp_path/'recovery/status.json').read_text())['phase']=='complete_predeclared_chain'


@pytest.mark.parametrize('failure',['predecessor','existing_output','changed_pin','command_failure','bad_result'])
def test_failures_do_not_retry_or_continue(tmp_path,monkeypatch,failure):
    c,target,result=fixture(tmp_path,monkeypatch);calls=[]
    if failure=='predecessor':
        (tmp_path/'predecessor/status.json').write_text(json.dumps(dict(phase='stopped_preserve_handles')))
    if failure=='existing_output':(tmp_path/'rloo').mkdir()
    if failure=='changed_pin':(tmp_path/'source.py').write_text('changed')
    def invoke(argv,**kw):
        calls.append(argv)
        if failure=='command_failure':raise subprocess.CalledProcessError(1,argv)
        target.mkdir();result.write_text(json.dumps(dict(phase='incomplete')))
    monkeypatch.setattr(successor.subprocess,'run',invoke)
    with pytest.raises((ValueError,subprocess.CalledProcessError)):
        successor.run('template','local')
    assert len(calls)==(1 if failure in ('command_failure','bad_result') else 0)


def test_pidfd_waits_exact_process_and_rejects_reused_identity():
    process=subprocess.Popen([sys.executable,'-c','import sys;sys.stdin.read()'],stdin=subprocess.PIPE)
    try:
        identity=successor.process_identity(process.pid)
        with pytest.raises(ValueError):
            successor.wait_exact(dict(pid=process.pid,**{**identity,'start_ticks':'wrong'}))
        process.stdin.close()
        successor.wait_exact(dict(pid=process.pid,**identity))
    finally:process.wait(timeout=10)


def test_unregistered_execution_refused():
    with pytest.raises(ValueError):successor.command(dict(module='arbitrary.module',arguments=[]))

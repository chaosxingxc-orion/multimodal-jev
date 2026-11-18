import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from tools.reports.finalize_context_macro_training import wait_driver


def test_exact_live_process_is_waited_without_polling(tmp_path):
    p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.5)'])
    proc=Path('/proc')/str(p.pid)
    record=dict(pid=p.pid,start_ticks=(proc/'stat').read_text().split(') ')[1].split()[19],
        cmdline_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest())
    try:
        with pytest.raises(ValueError,match='reused'):wait_driver(dict(record,start_ticks='wrong'),tmp_path/'terminal.json')
        wait_driver(record,tmp_path/'terminal.json')
        assert p.wait(timeout=1)==0
    finally:
        if p.poll() is None:p.terminate();p.wait()


def test_missing_driver_requires_actual_terminal(tmp_path,monkeypatch):
    import tools.reports.finalize_context_macro_training as m
    def missing(_):raise ProcessLookupError()
    monkeypatch.setattr(m.os,'pidfd_open',missing)
    with pytest.raises(ValueError,match='missing without terminal'):wait_driver({'pid':1},tmp_path/'terminal.json')
    (tmp_path/'terminal.json').write_text(json.dumps({'exit_code':1}))
    wait_driver({'pid':1},tmp_path/'terminal.json')


def test_legacy_repair_runs_after_terminal_and_before_collection(tmp_path,monkeypatch):
    import tools.reports.finalize_context_macro_training as m
    import tools.reports.repair_context_macro_artifact_access as access
    events=[]
    (tmp_path/'driver.json').write_text('{}')
    (tmp_path/'terminal.json').write_text('{"exit_code":0}')
    pkg=tmp_path/'package';pkg.mkdir()
    (pkg/'training.json').write_text('{"start_update":1,"stop_update":2}')
    monkeypatch.setattr(m,'check_paths',lambda *a:[])
    monkeypatch.setattr(m,'wait_driver',lambda *a:events.append('wait'))
    monkeypatch.setattr(access,'run',lambda c:events.append('repair'))
    def collect(*a,**kw):
        events.append('collect')
        return dict(status='audited',fresh_process_replay_verified=True)
    monkeypatch.setattr(m,'collect_and_audit',collect)
    monkeypatch.setattr(m,'summarize',lambda *a,**kw:{'summary':True})
    m.run(dict(operations_root=str(tmp_path),prepared=str(tmp_path)),repair_legacy_access=True)
    assert events==['wait','repair','collect']
    assert json.loads((tmp_path/'finalization-terminal.json').read_text())['fresh_process_replay_verified']

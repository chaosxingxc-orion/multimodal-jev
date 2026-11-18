import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.reports.finalize_context_infonce import wait_exact


def record(child):
    return dict(pid=child.pid,argv=[sys.executable,'-c','import time;time.sleep(30)'],
        process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])


def test_exact_wait_rejects_pid_reuse_without_stopping_driver(tmp_path):
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    try:
        r=record(child);r['process_start_ticks']='wrong'
        with pytest.raises(ValueError,match='identity'):wait_exact(r,tmp_path/'terminal.json')
        assert child.poll() is None
    finally:child.terminate();child.wait()


def test_missing_driver_requires_terminal(tmp_path):
    child=subprocess.Popen([sys.executable,'-c','pass']);child.wait()
    r=dict(pid=child.pid,argv=[],process_start_ticks='missing')
    with pytest.raises(ValueError,match='missing'):wait_exact(r,tmp_path/'terminal.json')
    (tmp_path/'terminal.json').write_text('{"exit_code":0}')
    wait_exact(r,tmp_path/'terminal.json')

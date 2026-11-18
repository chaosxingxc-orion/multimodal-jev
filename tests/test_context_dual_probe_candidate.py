import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.reports.prepare_context_dual_probe_candidate import rebind_argv
from tools.reports.prepare_context_batch_probe_candidate import command,ROOT
from tools.reports.run_context_dual_probe_candidate import runner_namespace


def test_command_rebind_changes_only_explicit_runtime_bindings():
    old='/shared/models/base';new='/team/models/base';root='/local/runtime/dual-v1'
    original=command(old)
    changed=rebind_argv(original,ROOT,root,old,new,'/team/driver',7,'dual-v1')
    assert original==command(old)
    assert '--network=none' in changed and '--pull=never' in changed
    assert old+':'+old+':ro' not in changed and new+':'+new+':ro' in changed
    assert '/team/driver:/usr/local/Ascend/driver:ro' in changed
    assert root+'/package:'+root+'/package:ro' in changed
    assert root+'/package/tools/reports/run_context_dual_probe_candidate.py' in changed
    assert all(ROOT not in s for s in changed)
    assert changed[changed.index('--name')+1]=='dual-v1'


def test_finite_runner_keeps_failure_before_second_process(tmp_path,monkeypatch):
    import multimodal_jev.local_runtime_storage as storage
    import multimodal_jev.local_checkpoint as checkpoint
    monkeypatch.setattr(storage,'require_training_storage',lambda _:None)
    monkeypatch.setattr(checkpoint,'check_paths',lambda _:None)
    ns=runner_namespace(Path(__file__).resolve().parents[1])
    package=tmp_path/'package';package.mkdir();(package/'FILES.json').write_text('{}')
    calls=[]
    def fail(argv,**kwargs):
        calls.append(argv);kwargs['stdout'].write('failure retained\n')
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(ns['subprocess'],'run',fail)
    output=tmp_path/'result'
    with pytest.raises(RuntimeError,match='serial probe failed'):ns['run'](package,output)
    assert len(calls)==1 and calls[0][-1]=='serial'
    assert calls[0][2].endswith('/context_dual_measurement_candidate.py')
    assert json.loads((output/'terminal.json').read_text())['exit_code']==1
    assert json.loads((output/'serial-exit.json').read_text())['exit_code']==7
    assert (output/'serial.log').read_text()=='failure retained\n'
    assert not (output/'batched-command.json').exists()


def test_runner_refuses_unpinned_source(tmp_path):
    path=tmp_path/'tools/reports/run_context_batch_probe_candidate.py'
    path.parent.mkdir(parents=True);path.write_text('raise RuntimeError("must not execute")')
    with pytest.raises(ValueError,match='source changed'):runner_namespace(tmp_path)

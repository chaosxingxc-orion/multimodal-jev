import hashlib
import json
from pathlib import Path

import pytest

from tools.reports.prepare_context_batch_probe_candidate import command, ROOT, IMAGE
from tools.reports.run_context_batch_probe_candidate import verify


def test_sealed_tree_rejects_changed_extra_and_symlink(tmp_path):
    p=tmp_path/'source.py';p.write_text('x=1\n')
    manifest={'source.py':hashlib.sha256(p.read_bytes()).hexdigest()}
    (tmp_path/'FILES.json').write_text(json.dumps(manifest))
    assert verify(tmp_path)==hashlib.sha256((tmp_path/'FILES.json').read_bytes()).hexdigest()
    p.write_text('x=2\n')
    with pytest.raises(ValueError):verify(tmp_path)
    p.write_text('x=1\n');(tmp_path/'extra').write_text('x')
    with pytest.raises(ValueError):verify(tmp_path)
    (tmp_path/'extra').unlink();(tmp_path/'link').symlink_to(p)
    with pytest.raises(ValueError):verify(tmp_path)


def test_explicit_device_readonly_model_and_local_hot_paths():
    model='/mnt/cephfs/xingchao/models/Qwen__Qwen3-ASR-0.6B-hf'
    argv=command(model)
    assert IMAGE in argv and '--pull=never' in argv and '--network=none' in argv
    devices=[argv[i+1] for i,x in enumerate(argv) if x=='--device']
    assert devices==['/dev/'+x+':/dev/'+x for x in ('davinci7','davinci_manager','devmm_svm','hisi_hdc')]
    mounts=[argv[i+1] for i,x in enumerate(argv) if x=='-v']
    assert model+':'+model+':ro' in mounts and ROOT+'/package:'+ROOT+'/package:ro' in mounts
    assert '--privileged' not in argv and 'TMPDIR=/jev-tmp' in argv


def test_failed_serial_subprocess_records_terminal_without_batched_retry(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import multimodal_jev.local_runtime_storage as storage
    import multimodal_jev.local_checkpoint as checkpoint
    import tools.reports.run_context_batch_probe_candidate as runner
    monkeypatch.setattr(storage,'require_training_storage',lambda p:None)
    monkeypatch.setattr(checkpoint,'check_paths',lambda p:None)
    package=tmp_path/'package';package.mkdir();(package/'FILES.json').write_text('{}')
    calls=[]
    def fail(argv,**kwargs):
        calls.append(argv);kwargs['stdout'].write('retained failure\n');return SimpleNamespace(returncode=7)
    monkeypatch.setattr(runner.subprocess,'run',fail)
    output=tmp_path/'output'
    with pytest.raises(RuntimeError,match='serial probe failed'):runner.run(package,output)
    assert len(calls)==1 and calls[0][-1]=='serial'
    assert json.loads((output/'serial-exit.json').read_text())['exit_code']==7
    assert json.loads((output/'terminal.json').read_text())['exit_code']==1
    assert (output/'serial.log').read_text()=='retained failure\n'
    assert not (output/'batched-command.json').exists()

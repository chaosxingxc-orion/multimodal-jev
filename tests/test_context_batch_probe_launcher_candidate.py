import hashlib
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace as NS

import pytest

from tools.reports import launch_context_batch_probe_candidate as m


def package(tmp_path):
    p=tmp_path/'prepared/package';p.mkdir(parents=True)
    (p/'inputs.pt').write_bytes(b'input')
    (p/'code.py').write_text('source')
    policy=dict(model='/mnt/cephfs/xingchao/models/Qwen__Qwen3-ASR-0.6B-hf',
                physical_devices=[7],max_batch_queries=4,max_batch_tokens=16384)
    pins={m.ROOT+'/package/'+x.name:m.sha(x) for x in p.iterdir()}
    config=dict(policy=policy,source_root=m.ROOT+'/package',inputs=m.ROOT+'/package/inputs.pt',
                input_sha256=m.sha(p/'inputs.pt'),pins=pins)
    (p/'config.json').write_text(json.dumps(config))
    (p/'FILES.json').write_text(json.dumps({x.name:m.sha(x) for x in p.iterdir()}))
    receipt=dict(status='sealed_not_launched',image=m.IMAGE,remote_root=m.ROOT,
                 training_dispatch_allowed=False,files_sha256=m.sha(p/'FILES.json'),argv=m.command(policy['model']))
    (p.parent/'PREPARED.json').write_text(json.dumps(receipt))
    return p.parent


def test_exact_collection_config_and_immutable_payload(tmp_path):
    prepared=package(tmp_path);m.validate(prepared)
    archive=tmp_path/'p.tar.gz';m.archive_package(prepared,archive)
    with tarfile.open(archive) as t:
        assert t.extractfile('config.json').read()==t.extractfile('package/config.json').read()
        assert t.extractfile('package/FILES.json').read()==(prepared/'package/FILES.json').read_bytes()
    (prepared/'package/code.py').write_text('changed')
    with pytest.raises(ValueError):m.validate(prepared)


def test_manifest_and_config_must_bind_same_complete_pins(tmp_path):
    prepared=package(tmp_path);p=prepared/'package';c=json.loads((p/'config.json').read_text())
    c['pins'].pop(m.ROOT+'/package/code.py');(p/'config.json').write_text(json.dumps(c))
    files=json.loads((p/'FILES.json').read_text());files['config.json']=m.sha(p/'config.json')
    (p/'FILES.json').write_text(json.dumps(files))
    r=json.loads((prepared/'PREPARED.json').read_text());r['files_sha256']=m.sha(p/'FILES.json')
    (prepared/'PREPARED.json').write_text(json.dumps(r))
    with pytest.raises(ValueError,match='complete runtime'):m.validate(prepared)


def test_priority_and_device_observation_fail_closed(tmp_path):
    p=tmp_path/'minds-context-arithmetic-v1-queue-status.json'
    p.write_text(json.dumps(dict(phase='waiting')))
    with pytest.raises(ValueError):m.check_priority(tmp_path)
    p.write_text(json.dumps(dict(phase='frozen_MINDS_collected_and_scored')));m.check_priority(tmp_path)
    with pytest.raises(ValueError):m.check_device_observation('No running processes found in NPU 6')
    m.check_device_observation('No running processes found in NPU 7')


def test_observation_error_waits_same_handle_never_recreates(monkeypatch):
    calls=[];sleeps=[]
    def remote(argv,bounded):
        calls.append((argv,bounded))
        return NS(returncode=255,stdout='',stderr='disconnect') if len(calls)==1 else NS(returncode=0,stdout='7\n')
    monkeypatch.setattr(m.time,'sleep',sleeps.append)
    assert m.wait_exact(remote,'fixed-container')==7
    assert calls==[(['docker','wait','fixed-container'],False)]*2
    assert sleeps==[60]*10


def test_existing_handle_file_never_overwritten(tmp_path):
    p=tmp_path/'launch.json';m.write_new(p,dict(container_id='original'))
    with pytest.raises(FileExistsError):m.write_new(p,dict(container_id='replacement'))
    assert json.loads(p.read_text())['container_id']=='original'

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

TOOLS=Path(__file__).resolve().parents[1]/'tools/jev'
sys.path.insert(0,str(TOOLS))
import queue_span_native_actor_npu as queue


def test_exact_container_timeout_does_not_become_completion(monkeypatch):
    def failed(_):raise TimeoutError('observation timed out')
    monkeypatch.setattr(queue,'inspect',failed)
    with pytest.raises(TimeoutError):queue.completed('exact','image')
    for status,code in [('exited',1),('dead',0),('created',0)]:
        monkeypatch.setattr(queue,'inspect',lambda _,s=status,c=code:{'exact':{'Image':'image','State':{'Status':s,'ExitCode':c}}})
        with pytest.raises(ValueError):queue.completed('exact','image')
    monkeypatch.setattr(queue,'inspect',lambda _:{'exact':{'Image':'image','State':{'Status':'running','ExitCode':0}}})
    assert queue.completed('exact','image') is False


def gate_fixture(tmp_path):
    def save(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    source=tmp_path/'source/x.py';source.parent.mkdir();source.write_text('pass\n')
    save(tmp_path/'source-manifest.json',{'x.py':queue.file_hash(source)})
    save(tmp_path/'config.json',dict(pins={},base_id='base',adapter_id='adapter'))
    records=[dict(id=str(i),request_id=str(i),request_identity={'i':i},candidate_logprobs=[[-0.6931471805599453]*2]) for i in range(44)]
    for backend in ('hf','vllm'):
        save(tmp_path/f'run/{backend}/result.json',dict(passed=True,base_id='base',adapter_id='adapter',records=records))
        save(tmp_path/f'run/{backend}/terminal.json',{'exit_code':0})
    save(tmp_path/'run/terminal.json',{'exit_code':0})
    save(tmp_path/'run/READY.json',dict(passed=True,requests=44,result_sha256=queue.file_hash(tmp_path/'run/vllm/result.json')))
    return dict(config_sha256=queue.file_hash(tmp_path/'config.json'),source_manifest_sha256=queue.file_hash(tmp_path/'source-manifest.json'))


def test_gate_recomputes_candidates_not_pass_flag(tmp_path):
    launch=gate_fixture(tmp_path)
    assert queue.verify_native_gate(tmp_path,launch)['passed']
    result=tmp_path/'run/vllm/result.json';value=json.loads(result.read_text())
    # The retained passed flag cannot hide a valid but different distribution.
    import math
    value['records'][0]['candidate_logprobs']=[[math.log(.8),math.log(.2)]]
    result.write_text(json.dumps(value))
    ready=tmp_path/'run/READY.json';r=json.loads(ready.read_text());r['result_sha256']=queue.file_hash(result);ready.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='parity failed'):queue.verify_native_gate(tmp_path,launch)


def test_existing_creation_attempt_is_never_relaunched(tmp_path,monkeypatch):
    (tmp_path/'create-attempt.json').write_text('{}')
    def forbidden(*a):raise AssertionError('must not contact Docker')
    monkeypatch.setattr(queue,'remote',forbidden);monkeypatch.setattr(queue,'check_devices',forbidden)
    with pytest.raises(ValueError,match='already claimed'):queue.launch(tmp_path,{},[])

import json
from pathlib import Path
import sys
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
import watch_omni_transfer_readiness as module


def test_dependency_requires_exact_live_process_or_successful_container(tmp_path,monkeypatch):
    launch=tmp_path/'evaluation-launch.json';launch.write_text('{"pid":123}')
    spec=dict(root=str(tmp_path),files={'evaluation-launch.json':file_hash(launch)},
        pid=123,process_start='birth',container_children=[''])
    monkeypatch.setattr(module,'process_start',lambda pid:'birth')
    assert module.observe_dependency(spec) is False
    monkeypatch.setattr(module,'process_start',lambda pid:None)
    with pytest.raises(RuntimeError,match='missing'):module.observe_dependency(spec)
    terminal=tmp_path/'evaluation-terminal.json';terminal.write_text('{"exit_code":1}')
    with pytest.raises(RuntimeError,match='failed'):module.observe_dependency(spec)
    terminal.write_text('{"exit_code":0}')
    (tmp_path/'launch.json').write_text('{"container_id":"cid","image":"image"}')
    item=dict(Id='cid',Image='image',State={'Status':'exited','ExitCode':0})
    monkeypatch.setattr(module,'inspect_container',lambda cid:item)
    assert module.observe_dependency(spec) is True
    item['State']['Status']='running'
    with pytest.raises(ValueError,match='successful'):module.observe_dependency(spec)


def test_estimate_uses_all_registered_requests_and_container_overhead(tmp_path):
    validation=tmp_path/'validation';run=validation/'run';run.mkdir(parents=True)
    (run/'identity.json').write_text('{"config":{"max_new_tokens":128}}')
    (run/'result.json').write_bytes(canonical(dict(requests=336,elapsed_seconds=772)))
    (run/'predictions.jsonl').write_bytes(b''.join(canonical({'seconds':2}) for _ in range(336)))
    queue=tmp_path/'queue';queue.mkdir()
    roster=[dict(prompt=p,shard=s) for p in ['parent','p1'] for s in range(136)]
    (queue/'queue-intent.json').write_bytes(canonical(dict(roster=roster,requests_per_prompt=2197104)))
    result=module.estimate(validation,queue)
    assert result['prompts']==2 and result['shard_container_count']==272
    assert result['observed_engine_and_hash_overhead_seconds']==100
    assert result['estimated_serial_wall_days']==(2*2197104*2+100*272)/86400
    assert not (queue/'dispatch-intent.json').exists()


def test_readiness_rejects_dispatch_mode(tmp_path):
    (tmp_path/'intent.json').write_text('{"dispatch":true}')
    with pytest.raises(ValueError,match='prepare-only'):module.execute(tmp_path)


def test_prepare_waits_for_both_dependencies_and_never_dispatches(tmp_path,monkeypatch):
    plan=tmp_path/'plan.json';plan.write_text('{}')
    validation=tmp_path/'validation';selection=tmp_path/'selection'
    queue=tmp_path/'queue';events=[]
    intent=dict(dispatch=False,coordinator_source=str(tmp_path),source_files={},
        dependencies={'selection':{'root':str(selection)},'train_validation':{'root':str(validation)}},
        shard_plan=str(plan),shard_plan_sha256=file_hash(plan),observation_seconds=100,
        queue_root=str(queue),native_probe=str(tmp_path/'native'),physical_devices=[6,7])
    (tmp_path/'intent.json').write_bytes(canonical(intent))
    checks=iter([False,True,True,True])
    monkeypatch.setattr(module,'observe_dependency',lambda spec:next(checks))
    monkeypatch.setattr(module.time,'sleep',lambda seconds:events.append('wait'))
    monkeypatch.setattr(module,'verify',lambda root:events.append('verify'))
    monkeypatch.setattr(module,'register',lambda *args:events.append('register'))
    def prepare(*args):
        assert (tmp_path/'preparation-intent.json').exists()
        events.append('prepare');return {'dispatched':False}
    monkeypatch.setattr(module,'prepare',prepare)
    monkeypatch.setattr(module,'estimate',lambda *args:{'estimate_is_not_budget_or_completion':True})
    module.execute(tmp_path)
    assert events==['wait','verify','register','prepare']
    assert json.loads((tmp_path/'progress.json').read_text())['phase']=='prepared_not_dispatched'
    assert not (queue/'dispatch-intent.json').exists()
    assert not (queue/'launch.json').exists()
    # Even a successful preparation cannot be replayed into the same root.
    monkeypatch.setattr(module,'observe_dependency',lambda spec:True)
    with pytest.raises(FileExistsError):module.execute(tmp_path)
    assert events==['wait','verify','register','prepare']


def test_missing_dependency_fails_before_preparation(tmp_path):
    (tmp_path/'intent.json').write_bytes(canonical(dict(dispatch=False,dependencies={})))
    with pytest.raises(ValueError,match='both'):module.execute(tmp_path)
    assert not (tmp_path/'preparation-intent.json').exists()

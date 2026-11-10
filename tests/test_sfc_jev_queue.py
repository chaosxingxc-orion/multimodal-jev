import json
from pathlib import Path
import sys
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from test_sfc_jev import load_tool
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))


def test_successful_predecessor_acceptance_then_single_dispatch_and_score(tmp_path,monkeypatch):
    module=load_tool('watch_sfc_jev_probe');predecessor=tmp_path/'native';predecessor.mkdir()
    for name in ('run/result.json','score/metrics.json'):
        path=predecessor/name;path.parent.mkdir();path.write_text('{}')
    (predecessor/'launch.json').write_text('{"container_id":"cid"}')
    (predecessor/'acceptance.json').write_bytes(canonical(dict(requests=6,source_groups=4,weight_updates=0,
        container_id='cid',result_sha256=file_hash(predecessor/'run/result.json'),score_sha256=file_hash(predecessor/'score/metrics.json'))))
    plan=dict(dependency_kind='native-sfc-train6',dependency={'root':str(predecessor)},automatic_retry=False)
    (tmp_path/'queue-plan.json').write_bytes(canonical(plan));(tmp_path/'config.json').write_text('{}')
    events=[];ready=iter([False,True])
    monkeypatch.setattr(module,'check_pins',lambda *a:events.append('pins'))
    monkeypatch.setattr(module,'observe_dependency',lambda spec:next(ready))
    monkeypatch.setattr(module.time,'sleep',lambda seconds:events.append('wait'))
    monkeypatch.setattr(module,'select_inputs',lambda cfg:events.append('inputs'))
    monkeypatch.setattr(module,'dispatch_once',lambda *a:events.append('dispatch'))
    monkeypatch.setattr(module,'score',lambda *a:events.append('score'))
    module.execute(tmp_path)
    assert events==['pins','wait','pins','inputs','dispatch','score']


@pytest.mark.parametrize('name',['run','dispatch-intent.json','dispatch.log','launch.json','evaluation-terminal.json'])
def test_no_automatic_replay(tmp_path,name):
    module=load_tool('watch_sfc_jev_probe');(tmp_path/name).touch()
    with pytest.raises(FileExistsError):module.execute(tmp_path)


def test_failed_dependency_never_dispatches(tmp_path,monkeypatch):
    module=load_tool('watch_sfc_jev_probe')
    (tmp_path/'queue-plan.json').write_bytes(canonical(dict(dependency_kind='native-sfc-train6',dependency={},automatic_retry=False)))
    monkeypatch.setattr(module,'check_pins',lambda *a:None)
    def failed(spec):raise RuntimeError('failed predecessor')
    monkeypatch.setattr(module,'observe_dependency',failed)
    monkeypatch.setattr(module,'dispatch_once',lambda *a:pytest.fail('must not dispatch'))
    with pytest.raises(RuntimeError,match='predecessor'):module.execute(tmp_path)


@pytest.mark.parametrize('seconds',[None,0,100000,True,7500.0])
def test_unbounded_dispatch_observation_rejected(tmp_path,seconds):
    from watch_sfc_omni_probe import dispatch_once
    (tmp_path/'preparation.json').write_bytes(canonical(dict(observation_seconds=seconds)))
    with pytest.raises(ValueError,match='bounded'):dispatch_once(tmp_path,{})

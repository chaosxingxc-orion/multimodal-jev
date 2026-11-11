import importlib.util,json
from pathlib import Path
import pytest
from multimodal_jev.baseline import file_hash

spec=importlib.util.spec_from_file_location('evaluation_controller',Path(__file__).resolve().parents[1]/'tools/jev/run_span_transfer_evaluation.py')
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)


def fixture(tmp_path):
    names=['initial-learning','final-learning','initial-selection','final-selection'];pins={}
    for name in names:
        p=tmp_path/name;p.mkdir();f=p/'config.json';f.write_text('{}');pins[str(f)]=file_hash(f)
    (tmp_path/'plan.json').write_text(json.dumps(dict(schema='jev-transfer-evaluation-plan-v1',stages=names,pins=pins)))
    return names


def test_all_fixed_stages_run_inference_then_verification(tmp_path,monkeypatch):
    names=fixture(tmp_path);calls=[]
    monkeypatch.setattr(controller.subprocess,'run',lambda command,**kwargs:calls.append(command))
    controller.run(tmp_path)
    assert len(calls)==8
    for i,name in enumerate(names):
        assert calls[2*i][1].endswith('infer_span_rule_baseline.py')
        assert str(tmp_path/name/'config.json') in calls[2*i]
        assert calls[2*i+1][1].endswith('verify_span_rule_baseline.py')
        assert str(tmp_path/name) in calls[2*i+1]


def test_failure_stops_before_later_stages(tmp_path,monkeypatch):
    fixture(tmp_path);calls=[]
    def child(command,**kwargs):
        calls.append(command)
        if len(calls)==2:raise RuntimeError('verification failed')
    monkeypatch.setattr(controller.subprocess,'run',child)
    with pytest.raises(RuntimeError):controller.run(tmp_path)
    assert len(calls)==2


@pytest.mark.parametrize('fault',['changed_config','wrong_order','missing_stage'])
def test_changed_plan_rejected_before_any_child(tmp_path,monkeypatch,fault):
    fixture(tmp_path);path=tmp_path/'plan.json';plan=json.loads(path.read_text())
    if fault=='changed_config':(tmp_path/'initial-learning/config.json').write_text('{"changed":true}')
    elif fault=='wrong_order':plan['stages'].reverse()
    else:plan['stages'].pop()
    path.write_text(json.dumps(plan));calls=[]
    monkeypatch.setattr(controller.subprocess,'run',lambda *a,**k:calls.append(a))
    with pytest.raises(ValueError):controller.run(tmp_path)
    assert calls==[]

import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from multimodal_jev.baseline import atomic_json,file_hash
import queue_intent_matched_report as module

def fixture(tmp_path):
    path=tmp_path/'comparison.json';atomic_json(path,dict(image='fixed',pins={},arms={a:[dict(suite_root=f'{a}{i}') for i in range(12)] for a in ('D','Q','P')}))
    out=tmp_path/'run';out.mkdir()
    return dict(comparison_config=str(path),image='fixed',predecessor_suite='P11',pins={str(path):file_hash(path)}),out

def test_failed_actual_predecessor_cannot_produce_report(tmp_path,monkeypatch):
    cfg,out=fixture(tmp_path)
    def fail(*a):raise ValueError('actual P12 failed')
    monkeypatch.setattr(module,'await_qp',fail)
    monkeypatch.setattr(module,'compare',lambda *a:pytest.fail('must not score'))
    with pytest.raises(ValueError,match='P12 failed'):module.run(cfg,out)

@pytest.mark.parametrize('complete',[False,True])
def test_only_full_curve_can_be_published(tmp_path,monkeypatch,complete):
    cfg,out=fixture(tmp_path);monkeypatch.setattr(module,'await_qp',lambda *a:None)
    result=dict(complete=True,role='learning',updates=12,recordings=89,curve=[dict(update=i) for i in range(13 if complete else 12)])
    monkeypatch.setattr(module,'compare',lambda *a:result)
    if complete:
        module.run(cfg,out);assert module.read(out/'comparison.json')==result
    else:
        with pytest.raises(ValueError,match='complete fixed'):module.run(cfg,out)

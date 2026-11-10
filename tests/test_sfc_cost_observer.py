from pathlib import Path
import subprocess
import sys
import pytest
from multimodal_jev.baseline import atomic_json,file_hash
from test_sfc_jev import load_tool

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))


def fixture(tmp_path):
    m=load_tool('watch_sfc_jev_cost');root=tmp_path/'observer';root.mkdir()
    populations={}
    for name in ('learning','selection','test_id','test_ood'):
        p=tmp_path/(name+'.json');p.write_text('{}');populations[name]=dict(path=str(p),sha256=file_hash(p))
    plan=dict(schema='sfc-jev-cost-observer-v1',dispatch=False,observation_seconds=110000,
        populations=populations,source_files={},dependency={'root':'/registered-train78'})
    atomic_json(root/'observer-plan.json',plan)
    return m,root,plan


def test_wait_transient_failure_then_estimate_all_populations(tmp_path,monkeypatch):
    m,root,_=fixture(tmp_path);events=[];calls=iter([False,'timeout',True])
    def observe(_):
        value=next(calls);events.append('poll')
        if value=='timeout':raise subprocess.TimeoutExpired('ssh',20)
        return value
    monkeypatch.setattr(m,'observe_dependency',observe);monkeypatch.setattr(m.time,'sleep',lambda _:None)
    def estimate(probe,path,pin):
        name=path.stem;events.append(name)
        return dict(split='train_id' if name in ('learning','selection') else name,
            partition={'part':name} if name in ('learning','selection') else None,
            population_decisions=100,shard_count=1,estimated_serial_wall_days=1,estimated_npu_device_hours=48,
            observed_min_sensitivity_wall_days=.5,observed_max_sensitivity_wall_days=2)
    monkeypatch.setattr(m,'estimate',estimate)
    m.execute(root)
    assert events==['poll','poll','poll','learning','selection','test_id','test_ood']
    assert m.json.loads((root/'result.json').read_text())['dispatched'] is False
    with pytest.raises(FileExistsError):m.execute(root)


def test_failed_predecessor_never_estimates(tmp_path,monkeypatch):
    m,root,_=fixture(tmp_path)
    def failed(_):raise RuntimeError('dependency failed')
    monkeypatch.setattr(m,'observe_dependency',failed)
    monkeypatch.setattr(m,'estimate',lambda *a:pytest.fail('must wait for acceptance'))
    with pytest.raises(RuntimeError,match='dependency'):m.execute(root)
    assert not (root/'estimates').exists()


def test_changed_population_refused_before_wait(tmp_path,monkeypatch):
    m,root,plan=fixture(tmp_path);Path(plan['populations']['learning']['path']).write_text('changed')
    monkeypatch.setattr(m,'observe_dependency',lambda _:pytest.fail('must reject changed plan'))
    with pytest.raises(ValueError,match='population'):m.execute(root)

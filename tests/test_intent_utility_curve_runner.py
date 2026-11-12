import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import run_intent_utility_curves as module
from multimodal_jev.baseline import atomic_json


def test_no_extension_before_actual_pilot_acceptance(tmp_path,monkeypatch):
    p=dict(arms=[dict(steps=[dict(training_root='t',suite_root='s')])])
    events=[]
    monkeypatch.setattr(module,'validate',lambda c:p)
    monkeypatch.setattr(module,'wait_tail',lambda *a:events.append('tail'))
    def fail(*a):raise ValueError('actual pilot not accepted')
    monkeypatch.setattr(module,'authenticate_pair',fail)
    monkeypatch.setattr(module,'verify_replay',lambda *a:pytest.fail('premature replay'))
    monkeypatch.setattr(module,'continue_step',lambda *a:pytest.fail('premature update'))
    with pytest.raises(ValueError,match='not accepted'):module.run({'image':'fixed'},tmp_path)
    assert events==['tail']


def test_replay_failure_blocks_every_extension(tmp_path,monkeypatch):
    p=dict(arms=[dict(steps=[{},{}]) for _ in range(3)]);events=[]
    monkeypatch.setattr(module,'validate',lambda c:p)
    monkeypatch.setattr(module,'wait_tail',lambda *a:None)
    monkeypatch.setattr(module,'authenticate_pair',lambda *a:events.append('accepted'))
    def fail(*a):raise ValueError('master mismatch')
    monkeypatch.setattr(module,'verify_replay',fail)
    monkeypatch.setattr(module,'continue_step',lambda *a:pytest.fail('premature update'))
    with pytest.raises(ValueError,match='master mismatch'):module.run({'image':'fixed'},tmp_path)
    assert len(events)==6


def test_all_arms_all_points_no_learning_score_selection(tmp_path,monkeypatch):
    arms=[]
    for name in ['correctness','smoothed-utility','uniform-credit']:
        steps=[]
        for i in range(12):
            t=tmp_path/f'{name}-t{i}';s=tmp_path/f'{name}-s{i}';s.mkdir()
            atomic_json(s/'launch.json',dict(container_id=f'{name}-{i}'))
            steps.append(dict(training_root=str(t),suite_root=str(s)))
        arms.append(dict(name=name,steps=steps))
    atomic_json(tmp_path/'protocol.json',{})
    events=[];monkeypatch.setattr(module,'validate',lambda c:dict(arms=arms))
    monkeypatch.setattr(module,'wait_tail',lambda *a:None)
    monkeypatch.setattr(module,'authenticate_pair',lambda *a:None)
    monkeypatch.setattr(module,'verify_replay',lambda *a:None)
    monkeypatch.setattr(module,'continue_step',lambda c,o:events.append(c['training_root']))
    # Zero accuracy for every arm must not alter the frozen budget.
    report=dict(reports={'D':{'metrics':{'final_accuracy':0.}}})
    monkeypatch.setattr(module,'compare',lambda *a:dict(full_before=copy.deepcopy(report),full_after=copy.deepcopy(report)))
    out=tmp_path/'out';out.mkdir()
    module.run(dict(image='fixed',pins={},lock_directory='locks',plan_root=str(tmp_path)),out)
    result=module.read(out/'curves.json')
    assert len(events)==30 and set(result['curves'])=={a['name'] for a in arms}
    assert all([p['update'] for p in v]==list(range(13)) for v in result['curves'].values())


def test_fsc_tail_delegates_to_full_curve_authentication(tmp_path,monkeypatch):
    import wait_intent_priority_tail as tail
    calls=[]
    monkeypatch.setattr(module,'authenticate_complete',lambda c,r:calls.append(r) or {'passed':True})
    assert tail.authenticate({'priority_kind':'utility_curves12'},tmp_path)=={'passed':True}
    assert calls==[tmp_path]


def test_fsc_rejects_incomplete_curve_without_authenticating_partial_scores(tmp_path,monkeypatch):
    from multimodal_jev.baseline import file_hash
    root=tmp_path/'controller';root.mkdir();(root/'run').mkdir()
    plan=tmp_path/'plan';plan.mkdir();pilot=tmp_path/'pilot.json';atomic_json(pilot,{})
    atomic_json(plan/'protocol.json',dict(pilot_protocol=str(pilot),arms=[{'name':'arm','steps':[]}]))
    atomic_json(root/'config.json',dict(image='fixed',plan_root=str(plan),pins={}))
    atomic_json(root/'run/curves.json',dict(protocol_sha256=file_hash(plan/'protocol.json'),role='learning',curves={'arm':[{'update':0}]}))
    monkeypatch.setattr(module,'validate_protocol',lambda *a:None)
    monkeypatch.setattr(module,'authenticate_pair',lambda *a:pytest.fail('partial curve accepted'))
    with pytest.raises(ValueError,match='all-arm'):module.authenticate_complete({'image':'fixed'},root)

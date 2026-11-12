import os
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import resume_intent_matched_pilots as module
from multimodal_jev.baseline import atomic_json,file_hash


def fixture(tmp_path,monkeypatch):
    old=tmp_path/'old';(old/'run/Q-update-0007').mkdir(parents=True)
    arms=[]
    for arm in ('Q','P'):
        steps=[]
        for i in range(12):
            tr=tmp_path/f'{arm}-train{i}';su=tmp_path/f'{arm}-suite{i}'
            tr.mkdir();su.mkdir();steps.append(dict(training_root=str(tr),suite_root=str(su)))
        arms.append(dict(arm=arm,steps=steps))
    cfg=dict(arms=arms,image='fixed',pins={})
    atomic_json(old/'config.json',cfg);atomic_json(old/'source-manifest.json',{})
    argv=Path(f'/proc/{os.getpid()}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    atomic_json(old/'launch.json',dict(pid=os.getpid(),argv=argv))
    atomic_json(old/'run/progress.json',dict(arm='Q',completed_updates=6,total_updates=12,accepted_suite=arms[0]['steps'][5]['suite_root']))
    stage=old/'run/Q-update-0007';atomic_json(stage/'progress.json',dict(stage='waiting_resource_before_first_create'))
    tr=Path(arms[0]['steps'][6]['training_root']);receipt=dict(container_id='exact',image='fixed',physical_devices=[4,5,6,7])
    atomic_json(tr/'launch.json',receipt);atomic_json(stage/'training-dispatched.json',receipt)
    cfg=dict(cfg,resume_from_controller=str(old),pins={str(old/n):file_hash(old/n) for n in ('config.json','launch.json','source-manifest.json')})
    monkeypatch.setattr(module,'validate',lambda config:None)
    monkeypatch.setattr(module,'completed',lambda cid,image:True)
    return cfg


def test_preflight_live_is_not_permission_to_take_over(tmp_path,monkeypatch):
    cfg=fixture(tmp_path,monkeypatch)
    assert module.boundary(cfg,require_stopped=False)==6
    atomic_json(Path(cfg['resume_from_controller'])/'superseded.json',dict(observed_process_exit=True,pid=os.getpid()))
    with pytest.raises(ValueError,match='must be stopped'):module.boundary(cfg)


@pytest.mark.parametrize('bad',['pending_suite','future_train','future_suite','running_training','dispatch','protocol'])
def test_claimed_or_changed_boundary_rejects_takeover(tmp_path,monkeypatch,bad):
    cfg=fixture(tmp_path,monkeypatch)
    if bad in ('pending_suite','future_train','future_suite'):
        index=6 if bad=='pending_suite' else 7
        key='training_root' if bad=='future_train' else 'suite_root'
        atomic_json(Path(cfg['arms'][0]['steps'][index][key])/'create-attempt.json',{})
    elif bad=='running_training':monkeypatch.setattr(module,'completed',lambda *a:False)
    elif bad=='dispatch':atomic_json(Path(cfg['resume_from_controller'])/'run/Q-update-0007/acceptance-dispatched.json',{})
    else:cfg['image']='changed'
    with pytest.raises(ValueError):module.boundary(cfg,require_stopped=False)


def test_stopped_original_allows_same_unclaimed_suite_only(tmp_path,monkeypatch):
    cfg=fixture(tmp_path,monkeypatch);old=Path(cfg['resume_from_controller'])
    atomic_json(old/'launch.json',dict(pid=999999999,argv=['missing']))
    cfg['pins'][str(old/'launch.json')]=file_hash(old/'launch.json')
    atomic_json(old/'superseded.json',dict(observed_process_exit=True,pid=999999999))
    assert module.boundary(cfg)==6

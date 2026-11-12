import os
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/jev'))
from multimodal_jev.baseline import atomic_json, file_hash
import wait_intent_priority_tail as tail
import run_fsc_transfer_matrix as fsc
import run_intent_binding_matrix as binding


def fixture(tmp_path):
    root = tmp_path/'prior'; root.mkdir(); (root/'run').mkdir()
    atomic_json(root/'config.json', {})
    atomic_json(root/'launch.json', dict(pid=os.getpid(), argv=Path(f'/proc/{os.getpid()}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')))
    return dict(priority_predecessor=str(root), pins={str(root/n):file_hash(root/n) for n in ('config.json','launch.json')})


def test_waits_600_and_refuses_failed_tail(tmp_path, monkeypatch):
    cfg = fixture(tmp_path); waits=[]
    def sleep(n):
        waits.append(n)
        atomic_json(Path(cfg['priority_predecessor'])/'run/terminal.json', dict(exit_code=1))
    monkeypatch.setattr(tail.time, 'sleep', sleep)
    with pytest.raises(ValueError, match='predecessor failed'):
        tail.wait_tail(cfg, tmp_path)
    assert waits == [600]


def test_missing_tail_never_counts_as_success(tmp_path):
    cfg=fixture(tmp_path); p=Path(cfg['priority_predecessor'])/'launch.json'
    atomic_json(p, dict(pid=999999999, argv=['missing'])); cfg['pins'][str(p)]=file_hash(p)
    with pytest.raises(ValueError, match='missing without terminal'):
        tail.wait_tail(cfg, tmp_path)


def test_fsc_cannot_prepare_before_priority_acceptance(tmp_path, monkeypatch):
    builder=tmp_path/'builder.json'; atomic_json(builder,{})
    cfg=dict(builder_config=str(builder),priority_predecessor='tail',pins={str(builder):file_hash(builder)})
    events=[]
    monkeypatch.setattr(fsc,'await_training',lambda *a:events.append('original-training'))
    def fail(*a):
        events.append('priority-tail'); raise ValueError('tail not accepted')
    monkeypatch.setattr(tail,'wait_tail',fail)
    monkeypatch.setattr(fsc,'prepare',lambda *a:pytest.fail('premature FSC preparation'))
    with pytest.raises(ValueError,match='tail not accepted'): fsc.run(cfg,tmp_path)
    assert events==['original-training','priority-tail']


def test_binding_uses_actual_lower_lr_training_gate(monkeypatch):
    called=[]
    monkeypatch.setattr(fsc,'await_training',lambda c,o:called.append((c,o)))
    cfg=dict(predecessor_kind='lower_lr_d12')
    binding.await_predecessor(cfg,'out')
    assert called==[(cfg,'out')]


def test_success_marker_without_actual_containers_is_rejected(tmp_path,monkeypatch):
    cfg=fixture(tmp_path);root=Path(cfg['priority_predecessor']);cfg['image']='image'
    training=tmp_path/'training';training.mkdir()
    atomic_json(root/'config.json',dict(image='image',pins={},training=str(training)))
    receipt=dict(image='image',container_id='cid')
    atomic_json(training/'launch.json',receipt);atomic_json(root/'run/training-dispatched.json',receipt)
    monkeypatch.setattr(tail,'completed',lambda *a:False)
    with pytest.raises(ValueError,match='actual priority completion'):
        tail.authenticate(cfg,root)


def test_complete_tail_authenticates_both_containers_and_pinned_result(tmp_path,monkeypatch):
    cfg=fixture(tmp_path);root=Path(cfg['priority_predecessor']);cfg['image']='image'
    paths={name:tmp_path/name for name in ('training','suite')}
    atomic_json(root/'config.json',dict(image='image',pins={},**{k:str(v) for k,v in paths.items()}))
    for name,target in paths.items():
        target.mkdir()
        for f in ('config.json','source-manifest.json'):atomic_json(target/f,{})
        receipt=dict(image='image',container_id=name,config_sha256=file_hash(target/'config.json'),source_manifest_sha256=file_hash(target/'source-manifest.json'))
        atomic_json(target/'launch.json',receipt);atomic_json(root/f'run/{name}-dispatched.json',receipt)
    suite=paths['suite']
    atomic_json(suite/'controller-terminal.json',dict(exit_code=0))
    atomic_json(suite/'verified.json',dict(passed=True,updates=2,ranks=[0,1,2,3],full_gate=dict(requests=178),adapter_id='adapter',files={}))
    accepted=dict(passed=True,adapter_id='adapter',verified_sha256=file_hash(suite/'verified.json'))
    atomic_json(suite/'ACCEPTED.json',accepted);atomic_json(root/'run/accepted.json',accepted)
    calls=[]
    def completed(cid,image):calls.append((cid,image));return True
    monkeypatch.setattr(tail,'completed',completed)
    assert tail.authenticate(cfg,root)['accepted']==accepted
    assert calls==[('training','image'),('suite','image')]
    atomic_json(suite/'verified.json',dict(passed=True))
    with pytest.raises(ValueError,match='independently accepted'):tail.authenticate(cfg,root)

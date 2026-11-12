from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from multimodal_jev.baseline import atomic_json,file_hash
import queue_intent_behavior_consumer as module


def test_cpu_command_has_no_devices_and_fixed_asset_mounts():
    root=Path('/mnt/cephfs/xingchao/experiments/consumer')
    argv=module.cpu_argv(root,'fixed',[dict(copy='/shared/asset',original='/home/credential')])
    assert '--device' not in argv and '--privileged' not in argv
    assert '/shared/asset:/home/credential:ro' in argv
    assert 'OMP_NUM_THREADS=2' in argv and '--network=none' in argv


def config(tmp_path):
    root=tmp_path/'consumer';root.mkdir();(root/'source').mkdir()
    ref=root/'reference.json';atomic_json(ref,{})
    atomic_json(root/'source-manifest.json',{})
    return dict(root=str(root),priority_kind='resident_policy_suite',reference=str(ref),external=[],pins={str(p):file_hash(p) for p in (ref,root/'source-manifest.json')})


def test_failed_resident_dependency_never_binds_or_launches(tmp_path,monkeypatch):
    cfg=config(tmp_path)
    def fail(*a):raise ValueError('resident failed')
    monkeypatch.setattr(module,'wait_tail',fail)
    monkeypatch.setattr(module,'bind',lambda *a:pytest.fail('cannot bind'))
    monkeypatch.setattr(module,'remote',lambda *a:pytest.fail('cannot launch'))
    with pytest.raises(ValueError,match='resident failed'):module.run(cfg,tmp_path)


@pytest.mark.parametrize('claimed',['consumer.json','consumer-run','create-attempt.json','container-launch.json'])
def test_never_retries_claimed_attempt(tmp_path,claimed):
    cfg=config(tmp_path);(Path(cfg['root'])/claimed).touch()
    with pytest.raises(ValueError,match='already claimed'):module.run(cfg,tmp_path)

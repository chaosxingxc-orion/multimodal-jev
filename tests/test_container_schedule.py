"""Transport scheduling tests use fixture artifacts, never claim real training."""
import importlib.util
import json
from pathlib import Path
import shutil
import pytest


def test_two_image_schedule_hands_off_new_checkpoint_and_parent(tmp_path,monkeypatch):
    from test_rl_preflight import fixture
    config=fixture(tmp_path,adapted=True)
    config['actor'].update(device='cpu',dtype='float32')
    root=tmp_path/'experiment';source=root/'source';(source/'tools/jev').mkdir(parents=True)
    (source/'src/multimodal_jev').mkdir(parents=True)
    repo=Path(__file__).parents[1]
    for name in ['run_container_audio_rl.py','run_audio_rl.py']:
        shutil.copy2(repo/'tools/jev'/name,source/'tools/jev'/name)
    path=source/'tools/jev/run_container_audio_rl.py'
    spec=importlib.util.spec_from_file_location('container_schedule',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    calls=[];parents=[];checkpoints=[];exports=[]
    class Executor:
        def __init__(self,*a):pass
        def execute(self,role,output,args,extra_mounts=()):
            calls.append(role);output.mkdir()
            if role=='export':
                assert extra_mounts and extra_mounts[0].endswith(':rw')
                checkpoint=str(output/'merged');exports.append(checkpoint)
                result={'checkpoint':checkpoint,'policy_id':'updated'}
            else:
                cfg=json.loads(Path(args[args.index('--config')+1]).read_text())
                if role=='rollout':checkpoints.append(cfg['checkpoint']);result={}
                else:parents.append(cfg['parent_run']);result={'updates':len(parents)}
            (output/'result.json').write_text(json.dumps(result));(output/'terminal.json').write_text('{"exit_code":0}')
    monkeypatch.setattr(m,'ContainerExecutor',Executor)
    monkeypatch.setattr(m,'checkpoint_identity',lambda p:('updated',{}))
    monkeypatch.setattr(m,'validate_adapter_export',lambda *a,**k:None)
    runtime={'shared_root':str(tmp_path)}
    m.run(config,runtime,root)
    assert calls==['rollout','actor','export']*2
    assert checkpoints==[config['initial_checkpoint'],exports[0]]
    assert parents==[None,str(root/'iteration-0000/actor')]
    assert json.loads((root/'result.json').read_text())['optimizer_updates']==2
    with pytest.raises(ValueError,match='runtime changed'):m.run(config,{**runtime,'changed':True},root)

import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import freeze_intent_finite_acceptance as module
from multimodal_jev.baseline import atomic_json,file_hash


def fixture(tmp_path,monkeypatch,status='exited',exit_code=0):
    training=tmp_path/'training';suite=tmp_path/'suite';training.mkdir();suite.mkdir()
    for root in (training,suite):
        (root/'source').mkdir();(root/'source/code.py').write_text('snapshot')
        atomic_json(root/'source-manifest.json',{'code.py':file_hash(root/'source/code.py')})
    atomic_json(training/'config.json',dict(pins={},model_path='/model',runtime_profile='profile',step=0,arm='D'))
    atomic_json(training/'launch.json',dict(image='image',physical_devices=[4,5,6,7],container_id='exact',
        config_sha256=file_hash(training/'config.json'),source_manifest_sha256=file_hash(training/'source-manifest.json')))
    atomic_json(suite/'config-template.json',dict(pins={},training_root=str(training),model='/model',runtime_profile='profile'))
    atomic_json(suite/'prepared.json',dict(image='image',
        config_template_sha256=file_hash(suite/'config-template.json'),source_manifest_sha256=file_hash(suite/'source-manifest.json'),
        create_argv=['docker','create','-v',f'{tmp_path}:{tmp_path}:ro']))
    for rank in range(4):
        out=training/'run'/f'rank-{rank}';out.mkdir(parents=True)
        atomic_json(out/'terminal.json',{'exit_code':0})
        atomic_json(out/'result.json',dict(passed=True,rank=rank,world_size=4,updates=1,adapter_id='adapter',arm='D'))
        for name in ('selection.json','before.json','after.json','optimizer-local.pt'):(out/name).write_text('pinned state')
    for name in ('selection.json','supervision.json'):(training/'run'/name).write_text('pinned inputs')
    monkeypatch.setattr(module,'inspect',lambda ids:{'exact':dict(Id='exact',Image='image',State=dict(Status=status,ExitCode=exit_code))})
    monkeypatch.setattr(module,'checkpoint_identity',lambda path:('base' if str(path)=='/model' else 'adapter',{}))
    return suite,training


@pytest.mark.parametrize('status,exit_code',[('running',0),('exited',1),('created',0)])
def test_no_freeze_before_authoritative_success(tmp_path,monkeypatch,status,exit_code):
    suite,_=fixture(tmp_path,monkeypatch,status,exit_code)
    with pytest.raises(ValueError,match='container must exit zero'):module.freeze(suite)
    assert not (suite/'config.json').exists()


def test_freeze_binds_actual_state_and_refuses_overwrite(tmp_path,monkeypatch):
    suite,training=fixture(tmp_path,monkeypatch)
    result=module.freeze(suite)
    assert result['status']=='frozen_not_dispatched'
    config=json.loads((suite/'config.json').read_text())
    for rank in range(4):
        path=training/'run'/f'rank-{rank}'/'optimizer-local.pt'
        assert config['pins'][str(path)]==file_hash(path)
    with pytest.raises(ValueError,match='already claimed'):module.freeze(suite)


def test_partial_rank_failure_cannot_freeze(tmp_path,monkeypatch):
    suite,training=fixture(tmp_path,monkeypatch)
    atomic_json(training/'run/rank-3/terminal.json',{'exit_code':1})
    with pytest.raises(ValueError,match='matching finite update'):module.freeze(suite)
    assert not (suite/'config.json').exists()

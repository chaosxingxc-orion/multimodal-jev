import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import freeze_intent_continuation as module
from multimodal_jev.baseline import atomic_json,file_hash
from test_intent_resume import fixture as predecessor


def fixture(tmp_path,monkeypatch,status='exited',code=0):
    cfg=predecessor(tmp_path);root=tmp_path/'next';root.mkdir()
    for base in (Path(cfg['previous_root']),Path(cfg['previous_suite']),root):
        (base/'source').mkdir();(base/'source/code.py').write_text('fixed source')
        atomic_json(base/'source-manifest.json',{'code.py':file_hash(base/'source/code.py')})
        if base!=root:
            data=json.loads((base/'config.json').read_text());data['pins']={};atomic_json(base/'config.json',data)
            cfg['pins'][str(base/'config.json')]=file_hash(base/'config.json')
            atomic_json(base/'launch.json',dict(container_id=base.name,image='image',physical_devices=[4] if base.name=='suite' else [4,5,6,7],
                config_sha256=file_hash(base/'config.json'),source_manifest_sha256=file_hash(base/'source-manifest.json')))
    suite=Path(cfg['previous_suite'])
    for name in ('READY.json','terminal.json','hf/result.json','hf/terminal.json','vllm/result.json','vllm/terminal.json'):
        path=suite/'run'/name;path.parent.mkdir(parents=True,exist_ok=True);atomic_json(path,{'exit_code':0})
    atomic_json(root/'config-template.json',cfg)
    atomic_json(root/'prepared.json',dict(image='image',config_template_sha256=file_hash(root/'config-template.json'),
        source_manifest_sha256=file_hash(root/'source-manifest.json'),create_argv=['docker','create','-v',f'{tmp_path}:{tmp_path}:ro']))
    monkeypatch.setattr(module,'inspect',lambda ids:{ids[0]:dict(Image='image',State=dict(Status=status,ExitCode=code))})
    monkeypatch.setattr(module,'checkpoint_identity',lambda path:('adapter',{}))
    return root


@pytest.mark.parametrize('status,code',[('running',0),('exited',1)])
def test_no_continuation_on_unfinished_or_failed_predecessor(tmp_path,monkeypatch,status,code):
    root=fixture(tmp_path,monkeypatch,status,code)
    with pytest.raises(ValueError,match='must exit zero'):module.freeze(root)
    assert not (root/'config.json').exists()


def test_frozen_continuation_binds_real_predecessor_artifacts(tmp_path,monkeypatch):
    root=fixture(tmp_path,monkeypatch)
    assert module.freeze(root)['status']=='frozen_not_dispatched'
    cfg=json.loads((root/'config.json').read_text())
    for rank in range(4):assert str(tmp_path/'training/run'/f'rank-{rank}'/'optimizer-local.pt') in cfg['pins']
    with pytest.raises(ValueError,match='already claimed'):module.freeze(root)

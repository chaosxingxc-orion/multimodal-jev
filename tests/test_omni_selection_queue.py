import importlib.util
import json
import sys
from pathlib import Path

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.omni_selection_contract import actor_config
from test_omni_selection_contract import fixture

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
spec=importlib.util.spec_from_file_location('omni_queue',Path(__file__).parents[1]/'tools/jev/run_omni_selection_queue.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_dispatch_preserves_devices_and_rebinds_visible_inputs(tmp_path):
    controls=tmp_path/'controls';parent=controls/'actual'
    root=tmp_path/'selection';child=root/'p1'
    argv=['docker','run','--name','controls-actual','--device','/dev/davinci4:/dev/davinci4',
          '--device','/dev/davinci5:/dev/davinci5','-v',str(parent)+':'+str(parent)+':rw',
          'image','timeout','--signal=TERM','--kill-after=30s','3600s','python3',
          str(parent/'source/tools/jev/infer_qwen3_omni_probe.py')]
    changed=module.selection_argv(argv,parent,child,controls,root,'p1')
    assert changed[changed.index('--name')+1]=='selection-p1'
    assert changed[-3]=='7200s'
    assert changed[-1]==str(child/'source/tools/jev/infer_qwen3_omni_selection.py')
    assert '/dev/davinci4:/dev/davinci4' in changed and '/dev/davinci5:/dev/davinci5' in changed
    assert str(child)+':'+str(child)+':rw' in changed
    assert all(str(parent) not in x for x in changed)
    assert '3600s' in argv


def test_queue_refuses_replay_before_any_new_dispatch(tmp_path,monkeypatch):
    protocol=tmp_path/'selection-protocol.json';protocol.write_text('{}')
    (tmp_path/'queue-intent.json').write_text(json.dumps(dict(order=['parent','p1'],protocol_sha256=file_hash(protocol))))
    for key in ['parent','p1']:(tmp_path/key).mkdir()
    (tmp_path/'p1/dispatch.log').write_text('ambiguous prior dispatch')
    def forbidden(*args):raise AssertionError('must not dispatch')
    monkeypatch.setattr(module,'dispatch_and_score',forbidden)
    with pytest.raises(FileExistsError,match='replay'):module.run(tmp_path)


def test_prepare_exports_snapshot_and_binds_all_inputs_without_dispatch(tmp_path,monkeypatch):
    cfg, _ = fixture(tmp_path)
    model=tmp_path/'model';model.mkdir()
    cfg['audio_model']=str(model)
    cfg['wall_seconds']=7200
    controls=tmp_path/'controls';parent=controls/'actual';parent.mkdir(parents=True)
    zeros=controls/'zeros';zeros.mkdir()
    (controls/'evaluation-terminal.json').write_text('{"exit_code":0}')
    for child in [parent,zeros]:
        (child/'run').mkdir();(child/'run/result.json').write_text('{"status":"passed"}')
    (parent/'config.json').write_text(json.dumps(cfg))
    argv=['docker','run','--name','controls-actual','--device','/dev/davinci4:/dev/davinci4',
          '--device','/dev/davinci5:/dev/davinci5',
          '-v',str(parent)+':'+str(parent)+':rw','-v',str(tmp_path)+':'+str(tmp_path)+':ro',
          'image','timeout','--signal=TERM','--kill-after=30s','3600s','python3',
          str(parent/'source/tools/jev/infer_qwen3_omni_probe.py')]
    (parent/'launch.json').write_text(json.dumps(dict(argv=argv,image='image',physical_devices=[4,5])))
    candidate=tmp_path/'candidate.txt';candidate.write_text('use exact schema')
    protocol_path=Path(cfg['selection_protocol']);protocol=json.loads(protocol_path.read_text())
    protocol['actor_config']=actor_config(cfg)
    protocol['prompts']['p1']=file_hash(candidate)
    for name in ['actual','zeros']:
        protocol['controls_'+name+'_result_sha256']=file_hash(controls/name/'run/result.json')
    protocol_path.write_text(json.dumps(protocol))
    def forbidden(*args):raise AssertionError('prepare must not dispatch')
    monkeypatch.setattr(module,'dispatch_and_score',forbidden)
    root=tmp_path/'selection'
    receipt=module.prepare(root,controls,protocol_path,{'parent':Path(cfg['instruction']),'p1':candidate},'HEAD')
    assert receipt['dispatched'] is False and receipt['order']==['parent','p1']
    for key in ['parent','p1']:
        child=root/key;intent=json.loads((child/'intent.json').read_text())
        config=json.loads((child/'config.json').read_text())
        assert file_hash(config['instruction'])==protocol['prompts'][key]
        assert config['selection_protocol_sha256']==file_hash(protocol_path)
        assert (child/'source/tools/jev/infer_qwen3_omni_selection.py').exists()
        assert intent['physical_devices']==[4,5] and intent['observation_seconds']==7500
        assert not (child/'dispatch.log').exists()
    assert file_hash(root/'parent/source/tools/jev/infer_qwen3_omni_probe.py')==file_hash(root/'p1/source/tools/jev/infer_qwen3_omni_probe.py')

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from multimodal_jev.baseline import file_hash

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

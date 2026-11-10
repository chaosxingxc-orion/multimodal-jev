import pytest
import json
from multimodal_jev.container_inputs import require_visible_inputs


def test_partition_must_be_present_inside_mount(tmp_path):
    run=tmp_path/'run';run.mkdir();local=run/'partition.json';local.write_text('{}')
    outside=tmp_path/'other.json';outside.write_text('{}')
    cfg={k:str(local) for k in ['requests','instruction','asset_audit','partition','audio_model']}
    argv=['docker','run','-v',str(run)+':'+str(run)+':ro']
    require_visible_inputs(argv,cfg)
    with pytest.raises(ValueError,match='outside'):require_visible_inputs(argv,{**cfg,'partition':str(outside)})
    local.unlink()
    with pytest.raises(ValueError,match='exist'):require_visible_inputs(argv,cfg)


def test_test_gate_requires_nested_selection_evidence_mount(tmp_path):
    run=tmp_path/'run';run.mkdir()
    local=run/'input';local.write_text('{}')
    selection=tmp_path/'selection.json';selection.write_text('{}')
    protocol=run/'test-protocol.json'
    protocol.write_text(json.dumps(dict(selection_result=str(selection),selection_protocol=str(selection))))
    cfg={k:str(local) for k in ['requests','instruction','asset_audit','audio_model']}
    cfg.update(evaluation_phase='test',test_protocol=str(protocol))
    argv=['docker','run','-v',str(run)+':'+str(run)+':rw']
    with pytest.raises(ValueError,match='outside'): require_visible_inputs(argv,cfg)
    argv.extend(['-v',str(selection)+':'+str(selection)+':ro'])
    require_visible_inputs(argv,cfg)

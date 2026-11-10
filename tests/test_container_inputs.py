import pytest
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

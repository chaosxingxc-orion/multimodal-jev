import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.container_pins import validate_bind_pins


def test_nested_mount_wins_and_remote_workspace_is_not_assumed(tmp_path):
    base=tmp_path/'base';base.mkdir();(base/'asset').write_text('wrong')
    exact=tmp_path/'exact';exact.write_text('right');digest=file_hash(exact)
    argv=['docker','create','-v',f'{base}:/runtime:ro','-v',f'{exact}:/runtime/asset:ro']
    assert validate_bind_pins(argv,{'/runtime/asset':digest})['passed']
    with pytest.raises(ValueError,match='missing/changed'):validate_bind_pins(argv[:4],{'/runtime/asset':digest})
    with pytest.raises(ValueError,match='not mounted'):validate_bind_pins(argv,{'/elsewhere/asset':digest})


def test_missing_mount_source_fails_before_launch(tmp_path):
    with pytest.raises(ValueError,match='missing/changed'):
        validate_bind_pins(['-v',f'{tmp_path}/missing:/workspace:ro'],{'/workspace/file':'a'*64})

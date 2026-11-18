import pytest
from tools.reports.repair_context_macro_artifact_access import repair


def fixture(tmp_path):
    for name in ('policy-0002','adapter-0002'):
        d=tmp_path/name;d.mkdir(mode=0o700)
        (d/'weights').write_bytes(b'unchanged tensors')
        (d/'weights').chmod(0o600)
    (tmp_path/'unrelated').write_bytes(b'keep private')
    (tmp_path/'unrelated').chmod(0o600)


def test_only_explicit_step_artifacts_change_modes(tmp_path):
    fixture(tmp_path)
    result=repair(tmp_path,[2])
    assert len(result['unchanged_files'])==2
    for name in ('policy-0002','adapter-0002'):
        assert (tmp_path/name).stat().st_mode & 0o777 == 0o755
        p=tmp_path/name/'weights'
        assert p.stat().st_mode & 0o777 == 0o644
        assert p.read_bytes()==b'unchanged tensors'
    assert (tmp_path/'unrelated').stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('fault',['missing','symlink','duplicate','invalid'])
def test_rejects_before_first_mutation(tmp_path,fault):
    fixture(tmp_path);steps=[2]
    if fault=='missing':steps=[2,3]
    if fault=='symlink':(tmp_path/'adapter-0002/link').symlink_to(tmp_path/'unrelated')
    if fault=='duplicate':steps=[2,2]
    if fault=='invalid':steps=[0]
    with pytest.raises(ValueError):repair(tmp_path,steps)
    assert (tmp_path/'policy-0002').stat().st_mode & 0o777 == 0o700

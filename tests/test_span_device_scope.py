import pytest
from multimodal_jev.span_device_scope import validate_tp1


@pytest.mark.parametrize('device', range(8))
def test_explicit_isolation_including_original_device(tmp_path, device):
    (tmp_path / f'davinci{device}').touch()
    (tmp_path / 'davinci_manager').touch()
    assert validate_tp1(dict(physical_devices=[device], max_model_len=4096),
                        device_root=tmp_path) == device


@pytest.mark.parametrize('devices', [[], [0, 1], [True], [-1], [8], ['0'], None])
def test_reject_implicit_or_invalid_device_scope(tmp_path, devices):
    with pytest.raises(ValueError):
        validate_tp1(dict(physical_devices=devices, max_model_len=4096), device_root=tmp_path)


@pytest.mark.parametrize('visible', [[], [4], [0, 4]])
def test_reject_missing_wrong_or_additional_visible_devices(tmp_path, visible):
    for device in visible:
        (tmp_path / f'davinci{device}').touch()
    with pytest.raises(ValueError):
        validate_tp1(dict(physical_devices=[0], max_model_len=4096), device_root=tmp_path)

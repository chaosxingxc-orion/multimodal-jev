"""Validate explicit physical device isolation before loading a TP1 model."""
from pathlib import Path


def validate_tp1(config, *, device_root=Path('/dev')):
    devices = config.get('physical_devices')
    if (not isinstance(devices, list) or len(devices) != 1
            or type(devices[0]) is not int or not 0 <= devices[0] <= 7
            or type(config.get('max_model_len')) is not int
            or config['max_model_len'] != 4096):
        raise ValueError('explicit single physical NPU0–7 and context4096 required')
    expected = [f'davinci{devices[0]}']
    if sorted(p.name for p in device_root.glob('davinci[0-9]*')) != expected:
        raise ValueError('visible physical NPU differs from isolated TP1 configuration')
    return devices[0]

"""Configurable TP1 process scope within a shared physical-device container.

Container device files alone do not establish runtime isolation. Query CANN
after process-local visibility was applied before importing torch/vLLM. The
mounted physical set and the visible ordinal come from the service policy, so
one module serves every declared topology instead of one per device layout.
"""
import os
from pathlib import Path


def validate_mapping(config, *, mounted, visible, count, physical):
    devices = config.get('physical_devices')
    scope = config.get('process_device_scope')
    if (not isinstance(devices, list) or len(devices) != 1 or type(devices[0]) is not int
            or config.get('max_model_len') != 4096):
        raise ValueError('explicit single-device TP1/context4096 required')
    if (not isinstance(scope, dict)
            or set(scope) != {'kind', 'mounted_physical', 'visible_ordinal'}
            or scope['kind'] != 'same_container_runtime_visibility_v1'
            or not isinstance(scope['mounted_physical'], list) or not scope['mounted_physical']
            or any(type(d) is not int or d < 0 for d in scope['mounted_physical'])
            or len(set(scope['mounted_physical'])) != len(scope['mounted_physical'])
            or not isinstance(scope['visible_ordinal'], str)
            or not scope['visible_ordinal'].isdigit()):
        raise ValueError('explicit versioned shared-container scope required')
    target = devices[0]; physical_set = scope['mounted_physical']
    if target not in physical_set:
        raise ValueError('service device is outside the declared mounted set')
    expected_visible = str(physical_set.index(target))
    if scope['visible_ordinal'] != expected_visible:
        raise ValueError('visible ordinal does not match the declared mounted set')
    if sorted(mounted) != sorted(f'davinci{d}' for d in physical_set):
        raise ValueError('mounted devices differ from declared shared scope')
    if visible != expected_visible or count != 1 or physical != target:
        raise ValueError('actual CANN process visibility/mapping differs')
    return target


def validate_tp1(config, *, device_root=Path('/dev')):
    # Reject static configuration before initializing a device context.
    mounted = [p.name for p in device_root.glob('davinci[0-9]*')]
    visible = os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
    target = validate_mapping(config, mounted=mounted, visible=visible, count=1,
                              physical=config['physical_devices'][0])
    import torch
    import torch_npu  # noqa: F401: installs the NPU backend
    from vllm.platforms import current_platform
    torch.npu.set_device(0)
    return validate_mapping(config, mounted=mounted, visible=visible,
                            count=torch.npu.device_count(),
                            physical=current_platform.visible_device_id_to_physical_device_id(0))

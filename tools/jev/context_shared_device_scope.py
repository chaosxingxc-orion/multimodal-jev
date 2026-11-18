"""Candidate TP1 process scope within a physical4–7 shared container.

Container device files alone do not establish runtime isolation. Query CANN
after process-local visibility was applied before importing torch/vLLM.
"""
import os
from pathlib import Path


def validate_mapping(config, *, mounted, visible, count, physical):
    if config.get('physical_devices') != [7] or config.get('max_model_len') != 4096:
        raise ValueError('explicit physical7 TP1/context4096 required')
    scope=config.get('process_device_scope')
    if scope != dict(kind='same_container_runtime_visibility_v1',
                     mounted_physical=[4,5,6,7], visible_ordinal='3'):
        raise ValueError('explicit versioned shared-container scope required')
    if sorted(mounted) != ['davinci4','davinci5','davinci6','davinci7']:
        raise ValueError('mounted devices differ from shared scope')
    if visible != '3' or count != 1 or physical != 7:
        raise ValueError('actual CANN process visibility/mapping differs')
    return 7


def validate_tp1(config, *, device_root=Path('/dev')):
    # Reject static configuration before initializing a device context.
    mounted=[p.name for p in device_root.glob('davinci[0-9]*')]
    visible=os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
    validate_mapping(config,mounted=mounted,visible=visible,count=1,physical=7)
    import torch
    import torch_npu  # noqa: F401: installs the NPU backend
    from vllm.platforms import current_platform
    torch.npu.set_device(0)
    return validate_mapping(config,mounted=mounted,visible=visible,
        count=torch.npu.device_count(),
        physical=current_platform.visible_device_id_to_physical_device_id(0))

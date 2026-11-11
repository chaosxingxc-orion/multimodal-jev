"""Read-only acoustic activation capture for a single-recording diagnostic.

Loaded by both the HF process and the trusted offline vLLM worker. Hooks never
return replacements. Snapshots clone storage so later in-place adds cannot
change the tensor being recorded, including on CPU.
"""
from pathlib import Path


def capture_first(module, name, destination, *, before=False):
    import torch
    path = Path(destination) / (name + '.pt')

    def save(value):
        if not path.exists():
            tensor = value[0] if isinstance(value, tuple) else value
            torch.save(tensor.detach().cpu().clone(), path)

    if before:
        def pre_hook(_module, inputs):
            save(inputs)
        return module.register_forward_pre_hook(pre_hook)

    def hook(_module, _inputs, output):
        save(output)
    return module.register_forward_hook(hook)


def install_audio_hooks(tower, projector, destination):
    import torch
    destination = Path(destination)
    handles = [capture_first(tower.conv2d1, 'conv_input', destination, before=True)]
    names = ['conv_input']
    for name in ('conv2d1', 'conv2d2', 'conv2d3', 'conv_out', 'ln_post'):
        handles.append(capture_first(getattr(tower, name), name, destination))
        names.append(name)
    handles.append(capture_first(tower.layers[0], 'encoder_input', destination, before=True))
    names.append('encoder_input')
    for index, layer in enumerate(tower.layers):
        name = f'layer_{index:02d}'
        handles.append(capture_first(layer, name, destination))
        names.append(name)
    handles.append(capture_first(projector, 'projected', destination))
    names.append('projected')
    torch.save(tower.positional_embedding.positional_embedding.detach().cpu().clone(),
               destination / 'position_buffer.pt')
    names.append('position_buffer')
    return handles, names


def install_engine_hooks(model, destination):
    # Qwen3-ASR's vLLM tower includes the multimodal projector.
    handles, names = install_audio_hooks(model.audio_tower, model.audio_tower, destination)
    # Keep handles alive for the diagnostic worker's lifetime.
    model._jev_audio_capture_handles = handles
    return {'tower_class': type(model.audio_tower).__module__ + '.' + type(model.audio_tower).__name__,
            'stages': names}

"""Read-only tensor hooks shared by HF and vLLM diagnostics."""

def attach(tower, projector):
    values = {'position_buffer': [tower.positional_embedding.positional_embedding.detach().cpu().clone()]}
    handles = []
    def capture(name, before=False):
        def hook(module, args, output=None):
            v = args[0] if before else output
            if isinstance(v, tuple): v = v[0]
            values.setdefault(name, []).append(v.detach().cpu().clone())
        return hook
    for name in ('conv2d1', 'conv2d2', 'conv2d3', 'conv_out', 'ln_post'):
        handles.append(getattr(tower, name).register_forward_hook(capture(name)))
    handles.append(tower.layers[0].register_forward_pre_hook(capture('encoder_input', True)))
    for i, layer in enumerate(tower.layers):
        handles.append(layer.register_forward_hook(capture(f'layer_{i:02d}')))
    handles.append(projector.register_forward_hook(capture('projected')))
    return handles, values


"""Locate the first divergent decoder operation on a fixed, label-free replay."""
import torch


@torch.inference_mode()
def trace(model, group, index, standard, reference):
    # Only materialize the selected row; do not store activations in receipts.
    states = {}
    checks = []
    handles = []
    lm = model.backbone.language_model
    width = group[index]['inputs']['input_ids'].numel()
    current = [0, True]
    def sample(x):
        if not torch.is_tensor(x) or x.ndim < 3:
            raise ValueError('expected batched decoder activation')
        return x[current[0], :width].detach().contiguous().cpu()
    def compare(key, x):
        v = sample(x)
        if current[1]:
            states[key] = v
        else:
            ref = states.pop(key)
            delta = (v.float() - ref.float()).abs()
            checks.append(dict(op=key, changed=int((delta != 0).sum()),
                               max_abs=float(delta.max())))
    def post(name):
        def see(mod, args, out):
            if isinstance(out, tuple):
                for i, value in enumerate(out):
                    if value is not None:
                        compare(f'{name}.{i}', value)
            else:
                compare(name, out)
        return see
    def pre(name):
        def see(mod, args):
            compare(name, args[0])
        return see
    layer = lm.layers[0]
    targets = [(lm.rotary_emb, 'rotary'), (layer.input_layernorm, 'input_norm'),
               (layer.self_attn.q_proj, 'q_proj'), (layer.self_attn.k_proj, 'k_proj'),
               (layer.self_attn.v_proj, 'v_proj'), (layer.self_attn.q_norm, 'q_norm'),
               (layer.self_attn.k_norm, 'k_norm'), (layer.self_attn.o_proj, 'o_proj'),
               (layer.post_attention_layernorm, 'post_attn_norm'), (layer.mlp, 'mlp')]
    # All-layer boundaries reveal divergence which first appears after layer zero.
    # Persist only aggregate differences, locally discarding the temporary tensors.
    for i, mod in enumerate(lm.layers):
        targets.append((mod, f'layer_{i}'))
    try:
        for module, name in targets:
            handles.append(module.register_forward_hook(post(name)))
        handles.append(layer.input_layernorm.register_forward_pre_hook(pre('layer_input')))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(pre('attention_output')))
        first = reference([group[index]])
        current[:] = [index, False]
        other = standard(group)
        if states or not checks:
            raise ValueError('incomplete paired activation trace')
        return dict(reference=float(first.item()), observed=float(other[index].item()),
                    checks=checks)
    finally:
        for handle in handles:
            handle.remove()

"""Read-only, comparable HF/vLLM language activations on full prefills.

Capture post-norm layer inputs (vLLM may carry a separate residual), all final
hidden states, and layer0 projections. No forward result is replaced. Captures
are CPU clones, so subsequent fused/in-place kernels cannot corrupt evidence.
"""
import torch


def attach(language, *, backend):
    if backend not in ('hf','vllm') or not language.layers:
        raise ValueError('explicit HF/vLLM language model with layers required')
    values={};handles=[]

    def record(name, value):
        if not isinstance(value,torch.Tensor) or not value.is_floating_point():
            raise ValueError('floating language activation required: '+name)
        values.setdefault(name,[]).append(value.detach().cpu().clone())

    def output(name):
        def hook(module,args,value):
            record(name,value[0] if isinstance(value,tuple) else value)
        return hook

    for index,layer in enumerate(language.layers):
        handles.append(layer.input_layernorm.register_forward_hook(output(f'layer_{index:02d}.input_norm')))
    handles.append(language.norm.register_forward_hook(output('final_norm')))
    layer=language.layers[0];attention=layer.self_attn
    handles.append(layer.post_attention_layernorm.register_forward_hook(output('layer_00.post_attention_norm')))
    handles.append(attention.o_proj.register_forward_hook(output('layer_00.attention_output')))
    handles.append(layer.mlp.down_proj.register_forward_hook(output('layer_00.mlp_output')))
    if backend=='hf':
        for name in ('q','k','v'):
            handles.append(getattr(attention,name+'_proj').register_forward_hook(output('layer_00.'+name)))
    else:
        def qkv(module,args,value):
            value=value[0] if isinstance(value,tuple) else value
            for name,part in zip(('q','k','v'),value.split(
                    [attention.q_size,attention.kv_size,attention.kv_size],-1),strict=True):
                record('layer_00.'+name,part)
        handles.append(attention.qkv_proj.register_forward_hook(qkv))
        def position_hook(module,args,kwargs):
            positions=kwargs.get('positions',args[0] if args else None)
            if not isinstance(positions,torch.Tensor) or positions.ndim!=1:
                raise ValueError('flat scheduler position trace required')
            values.setdefault('positions',[]).append(positions.detach().cpu().clone())
        handles.append(attention.register_forward_pre_hook(position_hook,with_kwargs=True))
    return handles,values


def compare(reference,candidate,*,lengths):
    """Compare actual ordered tensors; shape, missing stage and schedule fail closed."""
    from tools.reports.context_acoustic_numeric_probe import difference
    if (not reference or len(reference)!=len(lengths)
            or any(type(n) is not int or n<1 for n in lengths)):
        raise ValueError('complete positive sequence lengths and HF records required')
    names=set(reference[0])
    if any(set(row)!=names for row in reference) or set(candidate)!=names|{'positions'}:
        raise ValueError('complete same language stages required')
    positions=torch.cat(candidate['positions'])
    expected=torch.cat([torch.arange(n,dtype=positions.dtype) for n in lengths])
    if not torch.equal(positions,expected):
        raise ValueError('exact full-prefill scheduler sequence order required')
    def flat(parts):
        if not parts or any(p.ndim not in (2,3) for p in parts):
            raise ValueError('nonempty flat or single-row activations required')
        if any(p.ndim==3 and p.shape[0]!=1 for p in parts):
            raise ValueError('HF captures must be one complete request')
        return torch.cat([p.reshape(-1,p.shape[-1]) for p in parts])
    result={}
    # Compare all layer inputs before narrowing to layer0 projections. This does
    # not call a later norm mismatch proof of a specific preceding suboperation.
    for name in sorted(names):
        expected_parts=[]
        for row,length in zip(reference,lengths,strict=True):
            value=flat(row[name])
            if value.shape[0]!=length:raise ValueError('HF request length differs')
            expected_parts.append(value)
        a=torch.cat(expected_parts);b=flat(candidate[name])
        result[name]=difference(a,b)
    normalized=sorted(n for n in names if n.endswith('.input_norm'))+['final_norm']
    first=next((n for n in normalized if not result[n]['exact']),None)
    return dict(stages=result,first_normalized_hidden_difference=first,
        all_captured_stages_exact=all(r['exact'] for r in result.values()),
        full_model_parity_verified=False,
        scope='Read-only language activations; finite ordered batch diagnosis, not numerical admission or throughput')

"""Read-only first-layer norm, RoPE and attention-kernel boundary capture."""
import torch

from tools.reports.context_language_stage_capture import attach as language_attach


def attach(language, *, backend):
    handles,values=language_attach(language,backend=backend)
    attention=language.layers[0].self_attn

    def record(name,tensor):
        if not isinstance(tensor,torch.Tensor) or not tensor.is_floating_point():
            raise ValueError('floating attention boundary tensor required')
        # HF norm uses B,S,H,D; vLLM norm uses T,H,D. All captures are T,H*D.
        if tensor.ndim==4:
            if tensor.shape[0]!=1:raise ValueError('single complete HF request required')
            tensor=tensor[0]
        if tensor.ndim==3:tensor=tensor.reshape(tensor.shape[0],-1)
        if tensor.ndim!=2:raise ValueError('flat token-major attention boundary required')
        values.setdefault(name,[]).append(tensor.detach().cpu().clone())

    for kind in ('q','k'):
        def norm(module,args,result,kind=kind):record('layer_00.'+kind+'_norm',result)
        handles.append(getattr(attention,kind+'_norm').register_forward_hook(norm))
    def projected(module,args):record('layer_00.kernel_output',args[0])
    handles.append(attention.o_proj.register_forward_pre_hook(projected))
    if backend=='vllm':
        def rotary(module,args,result):
            record('layer_00.q_rope',result[0]);record('layer_00.k_rope',result[1])
        # Rotary may be shared by all layers. Only the call inside layer0 counts.
        active=[False]
        def begin(module,args,kwargs):active[0]=True
        def end(module,args,result):active[0]=False
        def filtered(module,args,result):
            if active[0]:rotary(module,args,result)
        handles.append(attention.register_forward_pre_hook(begin,with_kwargs=True))
        handles.append(attention.register_forward_hook(end,always_call=True))
        handles.append(attention.rotary_emb.register_forward_hook(filtered))
    else:
        import transformers.models.qwen3.modeling_qwen3 as qwen3
        original=qwen3.eager_attention_forward
        def captured(module,query,key,value,*args,**kwargs):
            if module is attention:
                record('layer_00.q_rope',query.transpose(1,2))
                record('layer_00.k_rope',key.transpose(1,2))
            return original(module,query,key,value,*args,**kwargs)
        qwen3.eager_attention_forward=captured
        class Restore:
            def remove(self):
                if qwen3.eager_attention_forward is not captured:
                    raise ValueError('HF attention capture owner changed')
                qwen3.eager_attention_forward=original
        handles.append(Restore())
    return handles,values

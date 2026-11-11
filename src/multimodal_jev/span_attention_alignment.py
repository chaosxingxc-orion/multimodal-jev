"""Diagnostic complete-prefill attention with HF operation order in vLLM.

Only one uncached full request, TP1, no decoding. Native projections, LoRA,
rotary and output projection remain in use. No prerecorded hidden states.
"""
import torch


def eager_causal_attention(q,k,v,scale):
    # T,H,D -> 1,H,T,D, as in HF Qwen3 eager attention.
    if (q.ndim!=3 or k.ndim!=3 or v.shape!=k.shape or q.shape[0]!=k.shape[0]
            or q.shape[2]!=k.shape[2] or q.shape[1]%k.shape[1]):
        raise ValueError('aligned full sequence grouped query attention required')
    q=q.transpose(0,1).unsqueeze(0)
    k=k.transpose(0,1).unsqueeze(0)
    v=v.transpose(0,1).unsqueeze(0)
    groups=q.shape[1]//k.shape[1]
    if groups!=1:
        k=k[:,:,None,:,:].expand(-1,-1,groups,-1,-1).reshape(1,q.shape[1],q.shape[2],q.shape[3])
        v=v[:,:,None,:,:].expand(-1,-1,groups,-1,-1).reshape_as(k)
    scores=torch.matmul(q,k.transpose(2,3))*scale
    mask=torch.full((q.shape[2],q.shape[2]),torch.finfo(q.dtype).min,
                    device=q.device,dtype=q.dtype).triu(1)
    scores=scores+mask
    probability=scores.softmax(-1,dtype=torch.float32).to(q.dtype)
    return torch.matmul(probability,v).transpose(1,2).contiguous().squeeze(0)


def aligned_attention_forward(module,positions,hidden_states):
    if (positions.ndim!=1 or hidden_states.ndim!=2 or len(positions)!=len(hidden_states)
            or not torch.equal(positions,torch.arange(len(positions),device=positions.device))):
        raise ValueError('single complete uncached sequence starting at zero required')
    qkv,_=module.qkv_proj(hidden_states)
    q,k,v=qkv.split([module.q_size,module.kv_size,module.kv_size],dim=-1)
    q=module.q_norm(q.view(len(q),module.num_heads,module.head_dim)).view(q.shape)
    k=module.k_norm(k.view(len(k),module.num_kv_heads,module.head_dim)).view(k.shape)
    q,k=module.rotary_emb(positions,q,k)
    out=eager_causal_attention(q.view(len(q),module.num_heads,module.head_dim),
                              k.view(len(k),module.num_kv_heads,module.head_dim),
                              v.view(len(v),module.num_kv_heads,module.head_dim),module.scaling)
    output,_=module.o_proj(out.reshape(len(hidden_states),module.q_size))
    return output


def install_eager_span_attention(model):
    from types import MethodType
    layers=model.language_model.model.layers
    for layer in layers:
        layer.self_attn.forward=MethodType(aligned_attention_forward,layer.self_attn)
    return {'layers':len(layers),'implementation':'HF-order causal eager attention; native QKV/LoRA/rotary',
            'scope':'TP1 single full uncached prefill only; no cache writes or decode support'}

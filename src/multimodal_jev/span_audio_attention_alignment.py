"""Diagnostic HF eager audio attention with explicit window boundaries."""
import torch


def windowed_audio_attention(q,k,v,boundaries,scale):
    if (q.ndim!=3 or k.shape!=q.shape or v.shape!=q.shape or boundaries.ndim!=1
            or boundaries.dtype not in (torch.int32,torch.int64)):
        raise ValueError('aligned T,H,D audio attention and integer boundaries required')
    limits=boundaries.detach().cpu().tolist()
    if len(limits)<2 or limits[0]!=0 or limits[-1]!=len(q) or any(a>=b for a,b in zip(limits,limits[1:])):
        raise ValueError('complete increasing audio windows required')
    result=[]
    for start,end in zip(limits,limits[1:]):
        query,key,value=[x[start:end].transpose(0,1).unsqueeze(0) for x in (q,k,v)]
        scores=torch.matmul(query,key.transpose(2,3))*scale
        probability=scores.softmax(-1,dtype=torch.float32).to(q.dtype)
        result.append(torch.matmul(probability,value).transpose(1,2).reshape(end-start,-1))
    return torch.cat(result,0)


def aligned_audio_attention_forward(module,hidden_states,cu_seqlens,max_seqlen=None):
    qkv,_=module.qkv_proj(hidden_states)
    q,k,v=[x.reshape(len(hidden_states),module.num_local_heads,module.head_dim) for x in qkv.chunk(3,-1)]
    attended=windowed_audio_attention(q,k,v,cu_seqlens,module.scaling)
    return module.out_proj(attended)[0]


def install_eager_audio_attention(model):
    from types import MethodType
    layers=model.audio_tower.layers
    for layer in layers:
        module=layer.self_attn
        if module.num_local_heads!=module.num_heads:
            raise ValueError('TP1 audio attention required')
    for layer in layers:
        layer.self_attn.forward=MethodType(aligned_audio_attention_forward,layer.self_attn)
    return dict(layers=len(layers),implementation='windowed HF eager noncausal audio attention',
                scope='Native audio QKV/output projections retained; no cross-window attention')

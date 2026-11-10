"""Opt-in diagnostic: omit keys excluded by every query in CPU text prefill.

No production actor default is changed. The scoped Transformers registry patch
is for a single-threaded frozen diagnostic process, not concurrent serving.
Query positions and valid attention edges remain unchanged. Gradients through
selected Q/K/V are preserved, but this is not training-readiness evidence.
"""
from contextlib import contextmanager


def compact_attention(original, module, query, key, value, attention_mask, *, records, **kwargs):
    import torch
    if attention_mask is None:
        records.append(dict(layer=module.layer_idx,mode='unchanged_no_mask',
                            query_length=query.shape[2],key_lengths=[key.shape[2]]*query.shape[0]))
        return original(module,query,key,value,attention_mask,**kwargs)
    if (query.ndim!=4 or key.ndim!=4 or value.ndim!=4 or attention_mask.ndim!=4
            or query.device.type!='cpu' or key.device!=query.device or value.device!=query.device
            or attention_mask.device!=query.device or attention_mask.dtype!=torch.bool
            or attention_mask.shape!=(query.shape[0],1,query.shape[2],key.shape[2])
            or query.shape[0]!=key.shape[0] or key.shape!=value.shape
            or query.shape[2]!=key.shape[2] or kwargs.get('dropout',0.)!=0.
            or kwargs.get('position_bias') is not None or kwargs.get('is_causal') is True):
        raise ValueError('CPU full prefill with explicit shared-head boolean mask and zero dropout required')
    outputs=[];lengths=[]
    for row in range(query.shape[0]):
        mask=attention_mask[row:row+1]
        keep=mask.any(dim=(0,1,2)).nonzero().flatten()
        if not keep.numel():raise ValueError('at least one reachable key required')
        selected=mask.index_select(-1,keep)
        # Restore the compact graph into its original coordinates as an
        # executable proof that no allowed edge was lost or introduced.
        restored=torch.zeros_like(mask);restored.index_copy_(-1,keep,selected)
        if not torch.equal(restored,mask):raise ValueError('attention graph changed')
        out,weights=original(module,query[row:row+1],key[row:row+1].index_select(2,keep),
                             value[row:row+1].index_select(2,keep),selected,**kwargs)
        if weights is not None:raise ValueError('SDPA diagnostic requires no attention weight output')
        outputs.append(out);lengths.append(len(keep))
    records.append(dict(layer=module.layer_idx,mode='compact_masked_keys',
                        query_length=query.shape[2],original_key_length=key.shape[2],key_lengths=lengths))
    return torch.cat(outputs,dim=0),None


@contextmanager
def compact_text_attention(hf):
    """Patch only Thinker text SDPA, restoring the registry on every exit."""
    registry=hf.ALL_ATTENTION_FUNCTIONS
    original=registry.get_interface('sdpa',hf.eager_attention_forward)
    if getattr(original,'_jev_compact_diagnostic',False):raise RuntimeError('nested diagnostic patch prohibited')
    records=[]
    def wrapped(module,query,key,value,attention_mask,**kwargs):
        if not isinstance(module,hf.Qwen3OmniMoeThinkerTextAttention):
            return original(module,query,key,value,attention_mask,**kwargs)
        return compact_attention(original,module,query,key,value,attention_mask,records=records,**kwargs)
    wrapped._jev_compact_diagnostic=True
    registry.register('sdpa',wrapped)
    try:yield records
    finally:registry.register('sdpa',original)

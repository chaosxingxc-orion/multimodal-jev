#!/usr/bin/env python3
"""Actual fixed HF mask/rotary and torch SDPA without model weights or audio."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path


def run():
    import torch
    import torch.nn.functional as F
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextRotaryEmbedding
    torch.set_num_threads(2); torch.manual_seed(17)
    cfg=Qwen3OmniMoeTextConfig(hidden_size=24,num_attention_heads=3,num_key_value_heads=1,
        rope_parameters={'rope_type':'default','rope_theta':1000000.,'mrope_section':[2,1,1]})
    cfg._attn_implementation='sdpa'
    rotary=Qwen3OmniMoeThinkerTextRotaryEmbedding(cfg)
    records=[]
    # First actual TRAIN prompt plus fixed 2 response tokens; independently
    # exercise small boundary cases. No real tokens or target labels consumed.
    for length,pad in ((7,3),(1076,154)):
        for dtype in (torch.float32,torch.bfloat16):
            masks=[]; positions=[]; rops=[]
            for padding in (0,pad):
                width=length+padding
                am=torch.ones(1,width,dtype=torch.long);am[:,:padding]=0
                pos=am.cumsum(-1)-1;pos.masked_fill_(am==0,1)
                embeds=torch.zeros(1,width,24,dtype=dtype)
                mask=create_causal_mask(cfg,embeds,am,None,position_ids=pos)
                masks.append(mask);positions.append(pos)
                rops.append(rotary(embeds,pos))
            raw, padded=masks
            logical=torch.ones(length,length,dtype=torch.bool).tril()
            if padded is None or padded.dtype!=torch.bool:
                raise ValueError('explicit boolean padded mask required')
            if not torch.equal(padded[0,0,pad:,pad:],logical) or padded[0,0,pad:,:pad].any():
                raise ValueError('valid tokens see different causal edges')
            if raw is not None and not torch.equal(raw[0,0],logical):
                raise ValueError('unpadded causal mask differs')
            if not torch.equal(positions[0],positions[1][:,pad:]):
                raise ValueError('valid positions differ')
            for a,b in zip(rops[0],rops[1]):
                torch.testing.assert_close(a,b[:,pad:],atol=0,rtol=0)
            q,k,v=[torch.randn(1,3,length,8).to(dtype) for _ in range(3)]
            qp,kp,vp=[F.pad(t,(0,0,pad,0)) for t in (q,k,v)]
            def attention(q,k,v,mask):
                return F.scaled_dot_product_attention(q,k,v,attn_mask=mask,
                    is_causal=mask is None,dropout_p=0.)
            implicit=attention(q,k,v,raw)
            explicit=attention(q,k,v,logical[None,None])
            expanded=attention(qp,kp,vp,padded)[:,:,pad:]
            # Counterfactual padded K/V values must not leak into valid outputs.
            kp[:,:,:pad]=100;vp[:,:,:pad]=-100
            changed=attention(qp,kp,vp,padded)[:,:,pad:]
            torch.testing.assert_close(expanded,changed,atol=0,rtol=0)
            records.append(dict(valid_length=length,left_padding=pad,dtype=str(dtype),
                unpadded_mask_is_none=raw is None,padded_mask_shape=list(padded.shape),
                valid_causal_edges_equal=True,valid_rotary_equal=True,
                padded_kv_counterfactual_equal=True,
                implicit_vs_explicit_max_error=float((implicit.float()-explicit.float()).abs().max()),
                unpadded_vs_padded_max_error=float((implicit.float()-expanded.float()).abs().max())))
    sources={inspect.getfile(obj):hashlib.sha256(Path(inspect.getfile(obj)).read_bytes()).hexdigest()
             for obj in (create_causal_mask,Qwen3OmniMoeThinkerTextRotaryEmbedding)}
    return dict(status='diagnostic_complete',records=records,source_sha256=sources,
        torch_version=torch.__version__,model_loaded=False,weight_updates=0,
        scope='Real HF mask/rotary and synthetic SDPA tensors at two shapes; no pretrained QKV, MoE, audio, full model, NPU or parity acceptance.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    try:
        result=run();(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    except BaseException as exc:
        (a.output/'terminal.json').write_text(json.dumps(dict(exit_code=1,error=repr(exc)))+'\n');raise
    (a.output/'terminal.json').write_text('{"exit_code": 0}\n')

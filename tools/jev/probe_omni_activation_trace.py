#!/usr/bin/env python3
"""Validate trace transparency on real tiny random Omni modules, no training."""
import argparse
import inspect
from pathlib import Path

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.omni_activation_trace import PaddingTrace


def run(out):
    import torch
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextModel
    torch.set_num_threads(2);torch.manual_seed(17)
    cfg=Qwen3OmniMoeTextConfig(vocab_size=64,hidden_size=24,intermediate_size=48,
        num_hidden_layers=2,num_attention_heads=3,num_key_value_heads=1,
        moe_intermediate_size=16,num_experts=4,num_experts_per_tok=2,
        max_position_embeddings=128,pad_token_id=0,
        rope_parameters={'rope_type':'default','rope_theta':1000000.,'mrope_section':[2,1,1]})
    cfg._attn_implementation='sdpa'
    model=Qwen3OmniMoeThinkerTextModel(cfg).to(torch.bfloat16).eval().requires_grad_(False)
    before={k:v.clone() for k,v in model.state_dict().items()}
    ids=torch.tensor([[7,8,9,10,11,12]])
    mask=torch.ones_like(ids);padded=torch.nn.functional.pad(ids,(3,0));pmask=padded.ne(0).long()
    def forward(ids,mask):
        pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,1)
        return model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False).last_hidden_state
    trace=PaddingTrace(out/'trace',max_bytes=1024**2)
    with torch.inference_mode():
        plain=forward(ids,mask);plain_padded=forward(padded,pmask)
        with trace.capture(model,mask):recorded=forward(ids,mask)
        with trace.capture(model,pmask,compare=True):compared=forward(padded,pmask)
        torch.testing.assert_close(plain,recorded,rtol=0,atol=0)
        torch.testing.assert_close(plain_padded,compared,rtol=0,atol=0)
        # Identity comparison should have no differing event.
        with trace.capture(model,mask,compare=True):forward(ids,mask)
        if any(not r['equal'] for r in trace.reports):raise ValueError('identity trace differs')
        # Change a valid token: trace must detect this, not compare only padding.
        changed=ids.clone();changed[0,0]=20
        with trace.capture(model,mask,compare=True):forward(changed,mask)
        if all(r['equal'] for r in trace.reports):raise ValueError('changed-token control not detected')
        negative_first=next(r['event'] for r in trace.reports if not r['equal'])
        with trace.capture(model,pmask,compare=True):forward(padded,pmask)
    if any(not torch.equal(v,before[k]) for k,v in model.state_dict().items()):
        raise ValueError('trace changed model state')
    if any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):
        raise ValueError('trace hooks leaked')
    return dict(status='passed',config=cfg.to_dict(),events=len(trace.reports),reports=trace.reports,
        reference=trace.reference,reference_tensor_bytes=trace.bytes,
        hook_transparency_exact=True,identity_comparison_exact=True,changed_token_first_event=negative_first,
        model_state_unchanged=True,hooks_removed=True,weight_updates=0,pretrained_weights_loaded=False,
        hf_source_sha256=file_hash(inspect.getfile(Qwen3OmniMoeThinkerTextModel)),
        tracer_sha256=file_hash(inspect.getfile(PaddingTrace)),
        scope='Real tiny random BF16 Omni SDPA/MoE module trace validation; no pretrained/audio/verl/NPU root-cause or parity claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    try:atomic_json(a.output/'result.json',run(a.output))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0))

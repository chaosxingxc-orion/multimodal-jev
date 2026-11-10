#!/usr/bin/env python3
"""CPU diagnostic of real Omni text attention; random tiny weights, no audio tower."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace


def run(actor_source):
    import torch
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextModel
    from verl.utils.torch_functional import logprobs_from_logits, logprobs_from_logits_v2
    torch.set_num_threads(2)
    torch.manual_seed(17)
    cfg=Qwen3OmniMoeTextConfig(vocab_size=64,hidden_size=24,intermediate_size=48,
        num_hidden_layers=2,num_attention_heads=3,num_key_value_heads=1,
        moe_intermediate_size=16,num_experts=4,num_experts_per_tok=2,
        max_position_embeddings=128,pad_token_id=0,
        rope_parameters={'rope_type':'default','rope_theta':1000000.,'mrope_section':[2,1,1]})
    cfg._attn_implementation='eager'
    model=Qwen3OmniMoeThinkerTextModel(cfg).eval()
    head=torch.nn.Linear(24,64,bias=False).eval()
    # Equal padded prompt width=5, response width=3; both padding directions.
    ids=torch.tensor([[0,0,7,8,9,10,11,12],[0,3,4,5,6,13,14,0]])
    mask=ids.ne(0).long();pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,1)
    response_mask=mask[:,-3:].bool()
    def logits(input_ids,attention_mask,position_ids):
        return head(model(input_ids=input_ids,attention_mask=attention_mask,
            position_ids=position_ids,use_cache=False).last_hidden_state)
    with torch.no_grad():
        reference=logits(ids,mask,pos)
        errors={}
        for axes in (3,4):
            value=logits(ids,mask,pos.unsqueeze(0).expand(axes,-1,-1))
            errors[str(axes)]=float((value[mask.bool()]-reference[mask.bool()]).abs().max())
            torch.testing.assert_close(value[mask.bool()],reference[mask.bool()],rtol=1e-5,atol=1e-6)
        # An independently unpadded single-row forward detects incorrect masking.
        for i in range(2):
            row=ids[i][mask[i].bool()].unsqueeze(0)
            single=logits(row,torch.ones_like(row),torch.arange(row.shape[1]).unsqueeze(0))
            torch.testing.assert_close(reference[i][mask[i].bool()],single[0],rtol=1e-5,atol=1e-6)
        changed=ids.clone();changed[:,6]=31
        altered=logits(changed,mask,pos)
        prefix=mask[:,:6].bool()
        torch.testing.assert_close(reference[:,:6][prefix],altered[:,:6][prefix],rtol=0,atol=0)

    # Execute the exact installed verl method body without actor initialization,
    # FSDP, device dispatch or optimizer. Globals are real torch and verl helper.
    tree=ast.parse(actor_source.read_text())
    owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DataParallelPPOActor')
    node=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='_forward_micro_batch')
    node.decorator_list=[]
    module=ast.Module(body=[node],type_ignores=[])
    # Record the actual installed dispatch; explicitly select its portable helper
    # only for this CPU diagnostic, never patch production worker globals.
    try:
        logprobs_from_logits(torch.zeros(1,64),torch.tensor([1]))
        dispatch_error=None
    except NotImplementedError as exc:
        dispatch_error=str(exc).split('\n')[0]
    scope={'torch':torch,'logprobs_from_logits':logprobs_from_logits_v2}
    exec(compile(ast.fix_missing_locations(module),str(actor_source),'exec'),scope)
    class Wrapper:
        def __call__(self,input_ids,attention_mask,position_ids,use_cache):
            assert use_cache is False
            return SimpleNamespace(logits=logits(input_ids,attention_mask,position_ids))
    actor=SimpleNamespace(config={},use_prefix_grouper=False,use_remove_padding=False,
        use_fused_kernels=False,device_name='cpu',param_dtype=torch.bfloat16,actor_module=Wrapper())
    batch=dict(input_ids=ids,attention_mask=mask,position_ids=pos,responses=ids[:,-3:])
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        independent=logits(ids,mask,pos).float().log_softmax(-1)
        expected=torch.stack([independent[i,4:7].gather(-1,ids[i,5:8,None]).squeeze(-1) for i in range(2)])
        observed=scope['_forward_micro_batch'](actor,batch,1.)['log_probs']
    error=float((observed[response_mask]-expected[response_mask]).abs().max())
    # BF16 helper may accumulate normalization at BF16 precision.
    torch.testing.assert_close(observed[response_mask].float(),expected[response_mask],rtol=0,atol=.04)
    return dict(scope='Random tiny Omni text MoE with real CPU eager attention; AST-extracted verl padded '
        'method. No audio encoder, pretrained weights, NPU, vLLM parity or training.',
        position_max_abs_error=errors,unpadded_matches=True,future_token_does_not_change_prefix=True,
        valid_response_tokens=int(response_mask.sum()),verl_logprob_max_abs_error=error,
        installed_logprob_dispatch_error=dispatch_error,diagnostic_logprob_helper='logprobs_from_logits_v2',
        dtype='FP32 position checks; BF16 actor autocast',seed=17,weight_updates=0,
        actor_source_sha256=hashlib.sha256(actor_source.read_bytes()).hexdigest(),
        hf_source_sha256=hashlib.sha256(Path(inspect.getfile(Qwen3OmniMoeThinkerTextModel)).read_bytes()).hexdigest(),
        config=cfg.to_dict())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--actor-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run(a.actor_source)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))

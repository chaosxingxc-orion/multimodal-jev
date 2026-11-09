#!/usr/bin/env python3
"""Real Qwen processor, masked CE/log-prob parity and one LoRA actor step."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from peft import LoraConfig, get_peft_model
from multimodal_jev.asr import load_runtime
from multimodal_jev.audio_policy import (teacher_forcing,load_audio,hf_prompt,vllm_request,
                                         response_log_probs,system_text)
from multimodal_jev.task_schema import Candidate,DecisionSchema,JevExample


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--audio',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(17)
    start=time.monotonic()
    processor,model=load_runtime(args.model,'cpu','float32')
    schema=DecisionSchema('smoke','noul','Does the speaker request a language change?')
    ex=JevExample('smoke','train',args.audio,schema,True)
    other=JevExample('smoke-negative','train',args.audio,
                     DecisionSchema('smoke','noul','Does the speaker request turning lights off?'),False)
    audio=load_audio(args.audio)
    one=hf_prompt(processor,ex,audio);two=hf_prompt(processor,other,audio)
    assert not torch.equal(one['input_ids'],two['input_ids'])
    assert torch.equal(one['input_features'],two['input_features'])
    assert torch.equal(one['input_features_mask'],two['input_features_mask'])
    request=vllm_request(processor,ex,audio)
    assert system_text(ex) in request['prompt'] and request['prompt'].count('<|audio_pad|>')==1
    inputs,boundary=teacher_forcing(processor,ex,audio)
    assert (inputs['labels'][:,:boundary]==-100).all()
    assert torch.equal(inputs['labels'][:,boundary:],inputs['input_ids'][:,boundary:])
    assert (inputs['input_ids']==model.config.audio_token_id).sum()==(one['input_ids']==model.config.audio_token_id).sum()
    lora=LoraConfig(r=4,lora_alpha=8,lora_dropout=0.0,
                    target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM',bias='none')
    # text-only projection names: audio tower q_proj/v_proj would otherwise also match.
    names=[name for name,module in model.named_modules()
           if name.startswith('model.language_model.') and name.rsplit('.',1)[-1] in {'q_proj','v_proj'}]
    lora.target_modules=names
    model=get_peft_model(model,lora)
    model.train()
    trainable=[param for param in model.parameters() if param.requires_grad]
    before=[param.detach().clone() for param in trainable]
    optimizer=torch.optim.AdamW(trainable,lr=1e-4,weight_decay=0,foreach=False,fused=False)
    output=model(**inputs,use_cache=False)
    logp=response_log_probs(output.logits,inputs['input_ids'],boundary)
    torch.testing.assert_close(-logp.mean(),output.loss,rtol=1e-5,atol=1e-5)
    output.loss.backward()
    grad_norm=float(torch.nn.utils.clip_grad_norm_(trainable,1.0))
    assert grad_norm>0 and torch.isfinite(torch.tensor(grad_norm))
    optimizer.step()
    changed=sum(not torch.equal(old,param.detach()) for old,param in zip(before,trainable))
    assert changed>0
    result={'status':'passed','prompt_tokens':boundary,'response_tokens':logp.shape[-1],
            'loss':float(output.loss.detach()),'grad_norm':grad_norm,'changed_parameter_tensors':changed,
            'trainable_parameters':sum(p.numel() for p in trainable),
            'schema_token_ids_differ':True,'audio_features_identical_across_schemas':True,
            'causal_logprob_ce_parity':True,'prompt_mask_verified':True,
            'vllm_request_only_one_unexpanded_audio_placeholder':True,
            'elapsed_seconds':time.monotonic()-start,
            'scope':'CPU single-example LoRA and request contract; no vLLM inference or verl training yet'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:json.dump(result,handle,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

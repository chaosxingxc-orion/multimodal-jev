#!/usr/bin/env python3
"""Actual verl KL gradients and Qwen frozen-reference update; controlled fixtures."""
import argparse
import copy
import json
from pathlib import Path
import time
import torch
from peft import LoraConfig,get_peft_model
from multimodal_jev.asr import load_runtime
from multimodal_jev.audio_policy import load_audio,hf_prompt,answer_ids,response_log_probs
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.reference_policy import reference_loss
from multimodal_jev.rl_update import update_actor
from multimodal_jev.rollout import Rollout
from multimodal_jev.task_schema import JevExample


def run(args):
    torch.set_num_threads(4);torch.manual_seed(17)
    # Known analytic gradient, masked nonfinite padding, and unequal-length
    # aggregation check against the actual pinned verl implementation.
    x=torch.tensor([[-1.,-2.,float('nan')],[-3.,float('inf'),float('nan')]],requires_grad=True)
    ref=torch.tensor([[-1.2,-1.5,float('nan')],[-2.8,float('nan'),float('inf')]])
    mask=torch.tensor([[1,1,0],[1,0,0]])
    loss=reference_loss(x,ref,mask,'k3');loss.backward()
    expected=torch.zeros_like(x);valid=mask.bool()
    expected[valid]=(1-torch.exp(ref[valid]-x.detach()[valid]))/3
    torch.testing.assert_close(x.grad,expected,atol=1e-7,rtol=1e-6)
    assert x.grad[~valid].eq(0).all() and torch.isfinite(loss)
    split=sum(reference_loss(x.detach()[i:i+1],ref[i:i+1],mask[i:i+1],'k3')*mask[i].sum()/3 for i in range(2))
    torch.testing.assert_close(loss.detach(),split)
    try:reference_loss(x,ref.clone().requires_grad_(),mask,'k3')
    except ValueError:pass
    else:raise AssertionError('trainable reference accepted')
    analytic={'loss':float(loss.detach()),'gradient_max_error':float((x.grad-expected).abs().max()),'masked_gradient_zero':True}
    atomic_json(args.output/'analytic.json',analytic)
    with args.manifest.open() as handle:
        ex=next(JevExample.from_record(row) for line in handle if (row:=json.loads(line))['metadata']['variant']=='noul.True')
    assert ex.split=='train' and file_hash(ex.audio)==ex.metadata['audio_sha256']
    processor,base=load_runtime(args.model,'cpu','float32')
    reference=copy.deepcopy(base).requires_grad_(False).eval()
    modules=[n for n,_ in base.named_modules() if n.startswith('model.language_model.') and n.rsplit('.',1)[-1] in {'q_proj','v_proj'}]
    model=get_peft_model(base,LoraConfig(r=4,lora_alpha=8,lora_dropout=0.,target_modules=modules,task_type='CAUSAL_LM',bias='none')).eval()
    # Deliberately move the actor away from its fixed reference, without
    # training or pretending that this is a sampled trajectory.
    with torch.no_grad():
        for n,p in model.named_parameters():
            if 'lora_B' in n:p.normal_(0.,0.01)
    params=[p for p in model.parameters() if p.requires_grad]
    before=[p.detach().clone() for p in params]
    wave=load_audio(ex.audio);ids=answer_ids(processor.tokenizer,True)
    inputs=dict(hf_prompt(processor,ex,wave));boundary=inputs['input_ids'].shape[-1]
    inputs['input_ids']=torch.cat((inputs['input_ids'],torch.tensor([ids])),dim=-1)
    inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
    with torch.no_grad():
        logp=response_log_probs(model(**inputs,use_cache=False).logits,inputs['input_ids'],boundary)[0]
        frozen_before=response_log_probs(reference(**inputs,use_cache=False).logits,inputs['input_ids'],boundary).clone()
    versions=[p._version for p in reference.parameters()]
    records=[Rollout(ex.example_id,i,'controlled-fixture',tuple(ids),tuple(logp.tolist()),processor.tokenizer.decode(ids,skip_special_tokens=True),'stop') for i in range(2)]
    optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=0.,foreach=False,fused=False)
    atomic_json(args.output/'progress.json',{'phase':'kl_only_update','heartbeat_unix':time.time()})
    metrics=update_actor(model,processor,optimizer,[ex],[wave],records,'controlled-fixture',2,
        max_logprob_error=3e-4,reference_model=reference,kl_coefficient=0.1,kl_estimator='k3')
    assert metrics['updated'] and metrics['rewards'][0]==metrics['rewards'][1]
    assert metrics['metrics']['actor/reference_kl']>0 and abs(metrics['metrics']['actor/policy_loss'])<1e-7
    changed=sum(not torch.equal(old,p) for old,p in zip(before,params));assert changed>0
    assert versions==[p._version for p in reference.parameters()]
    assert all(p.grad is None and not p.requires_grad for p in reference.parameters())
    with torch.no_grad():
        frozen_after=response_log_probs(reference(**inputs,use_cache=False).logits,inputs['input_ids'],boundary)
    torch.testing.assert_close(frozen_before,frozen_after,atol=0,rtol=0)
    return {'status':'passed','analytic':analytic,'metrics':metrics,'changed_parameter_tensors':changed,
        'reference_unchanged':True,'reference_gradient_absent':True,
        'source_sha256':{n:file_hash(Path(__file__).parents[2]/'src/multimodal_jev'/n) for n in ('reference_policy.py','rl_update.py','verl_audio.py')},
        'scope':'controlled equal-reward TRAIN fixtures, actual verl KL gradient and Qwen update; NOT vLLM rollout or efficacy'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:atomic_json(args.output/'result.json',run(args))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()});raise
    atomic_json(args.output/'terminal.json',{'exit_code':0,'finished_unix':time.time()})

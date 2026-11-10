#!/usr/bin/env python3
"""Score saved real vLLM responses with the identical merged HF checkpoint.

Read-only per-response diagnosis. No sampling, gradients, adapter or optimizer.
Completes all responses before reporting failure; never relaxes the actor gate.
"""
import argparse
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.rl_run import load_rollout_run


def cached_response_logprobs(model,inputs,boundary):
    """Teacher-force the saved response one token at a time with a KV cache.

    Diagnostic for prefill/decode shape sensitivity; never sample replacements.
    Acoustic features enter only the original prefill, as in online decoding.
    """
    import torch
    prompt={key:(value[:,:boundary] if key in {'input_ids','attention_mask','position_ids'} else value)
            for key,value in inputs.items()}
    output=model(**prompt,use_cache=True)
    values=[]
    response=inputs['input_ids'][:,boundary:]
    for index in range(response.shape[-1]):
        values.append(torch.log_softmax(output.logits[:,-1].float(),dim=-1)
                      .gather(-1,response[:,index:index+1]))
        if index+1<response.shape[-1]:
            output=model(input_ids=response[:,index:index+1],
                attention_mask=inputs['attention_mask'][:,:boundary+index+1],
                position_ids=inputs['position_ids'][:,boundary+index:boundary+index+1],
                past_key_values=output.past_key_values,use_cache=True)
    return torch.cat(values,dim=-1)


def run(args):
    import torch
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.audio_policy import hf_prompt,load_audio,response_log_probs
    identity,examples,records=load_rollout_run(args.rollout)
    torch.set_num_threads(4);torch.manual_seed(17)
    device=getattr(args,'device','cpu');dtype=getattr(args,'dtype','float32')
    processor,model=load_runtime(identity['config']['checkpoint'],device,dtype)
    by_id={ex.example_id:ex for ex in examples};results=[]
    for index,record in enumerate(records):
        atomic_json(args.output/'progress.json',{'phase':'forward','done':index,'total':len(records),'heartbeat_unix':time.time()})
        ex=by_id[record.example_id]
        inputs=dict(hf_prompt(processor,ex,load_audio(ex.audio)))
        boundary=inputs['input_ids'].shape[-1]
        inputs['input_ids']=torch.cat((inputs['input_ids'],torch.tensor([record.token_ids])),dim=-1)
        inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
        inputs['position_ids']=torch.arange(inputs['input_ids'].shape[-1]).unsqueeze(0)
        inputs={key:value.to(device=device,dtype=model.dtype if value.is_floating_point() else value.dtype)
                for key,value in inputs.items()}
        start=time.monotonic()
        with torch.inference_mode():
            if getattr(args,'cached_decode',False):
                logp=cached_response_logprobs(model,inputs,boundary)[0]
            else:
                logp=response_log_probs(model(**inputs,use_cache=False).logits,inputs['input_ids'],boundary)[0]
        old=torch.tensor(record.log_probs,device=logp.device);diff=(logp-old).abs()
        row={'example_id':ex.example_id,'replicate':record.replicate,'token_ids':record.token_ids,
             'vllm_logprobs':record.log_probs,'hf_logprobs':logp.tolist(),'max_error':float(diff.max()),
             'finite':bool(torch.isfinite(logp).all()),'seconds':time.monotonic()-start}
        results.append(row);atomic_json(args.output/f'response-{index:03d}.json',row)
    max_error=max(r['max_error'] for r in results)
    passed=all(r['finite'] and r['max_error']<=0.02 for r in results)
    return {'status':'passed' if passed else 'failed','max_logprob_error':max_error,'threshold':0.02,
            'responses':len(results),'policy_id':identity['policy_id'],'device':device,'dtype':dtype,
            'cached_decode':getattr(args,'cached_decode',False),
            'rollout_sha256':file_hash(args.rollout/'rollouts.jsonl'),
            'scope':'same merged weights HF single-response probability diagnosis; no optimizer or full-batch parity proof'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rollout',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--dtype',choices=['float32','bfloat16','float16'],default='float32')
    p.add_argument('--cached-decode',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        result=run(args);atomic_json(args.output/'result.json',result)
        if result['status']!='passed':raise ValueError('HF/vLLM response parity failed; per-response evidence retained')
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()});raise
    atomic_json(args.output/'terminal.json',{'exit_code':0,'finished_unix':time.time()})

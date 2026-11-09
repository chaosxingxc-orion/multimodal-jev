#!/usr/bin/env python3
"""Real HF actor + verl update with deliberately controlled response fixtures.

Never emits a vLLM rollout receipt. This checks the update implementation only.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import torch
from peft import LoraConfig,get_peft_model
from multimodal_jev.asr import load_runtime
from multimodal_jev.audio_policy import load_audio,hf_prompt,answer_ids,response_log_probs
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.rollout import Rollout
from multimodal_jev.rl_update import update_actor
from multimodal_jev.task_schema import JevExample


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:
        torch.set_num_threads(4);torch.manual_seed(17);start=time.monotonic()
        with args.manifest.open() as handle:
            ex=next(JevExample.from_record(row) for line in handle
                    if (row:=json.loads(line))['metadata']['variant']=='noul.True')
        if ex.split!='train':raise ValueError('TRAIN only')
        wave=load_audio(ex.audio)
        processor,base=load_runtime(args.model,'cpu','float32')
        modules=[name for name,_ in base.named_modules() if name.startswith('model.language_model.') and name.rsplit('.',1)[-1] in {'q_proj','v_proj'}]
        model=get_peft_model(base,LoraConfig(r=4,lora_alpha=8,lora_dropout=0.,target_modules=modules,task_type='CAUSAL_LM',bias='none')).eval()
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,foreach=False,fused=False)
        records=[]
        atomic_json(args.output/'progress.json',{'phase':'fixture_logprobs','heartbeat_unix':time.time()})
        for i,value in enumerate((True,False)):
            ids=answer_ids(processor.tokenizer,value)
            batch=dict(hf_prompt(processor,ex,wave));boundary=batch['input_ids'].shape[-1]
            batch['input_ids']=torch.cat((batch['input_ids'],torch.tensor([ids])),dim=-1)
            batch['attention_mask']=torch.ones_like(batch['input_ids'])
            with torch.inference_mode():
                logp=response_log_probs(model(**batch,use_cache=False).logits,batch['input_ids'],boundary)[0].tolist()
            records.append(Rollout(ex.example_id,i,'controlled-fixture',tuple(ids),tuple(logp),
                                   processor.tokenizer.decode(ids,skip_special_tokens=True),'stop'))
        parameters=[p for p in model.parameters() if p.requires_grad]
        before=[p.detach().clone() for p in parameters]
        try:
            update_actor(model,processor,optimizer,[ex],[wave],
                         [replace(records[0],text='{"value":false}'),records[1]],'controlled-fixture',2)
        except ValueError as exc:
            if 'reward text' not in str(exc):raise
        else:raise AssertionError('text/token mismatch accepted')
        atomic_json(args.output/'progress.json',{'phase':'update','heartbeat_unix':time.time()})
        metrics=update_actor(model,processor,optimizer,[ex],[wave],records,'controlled-fixture',2,max_logprob_error=3e-4)
        changed=sum(not torch.equal(old,p) for old,p in zip(before,parameters))
        assert metrics['updated'] and changed>0
        assert optimizer.state
        # Save real state for inspection, but never mark fixture trajectories as on-policy samples.
        model.save_pretrained(args.output/'adapter')
        torch.save(optimizer.state_dict(),args.output/'optimizer.pt')
        report={'status':'passed','metrics':metrics,'changed_parameter_tensors':changed,
                'text_token_mismatch_rejected':True,'optimizer_states':len(optimizer.state),
                'elapsed_seconds':time.monotonic()-start,
                'update_code_sha256':file_hash(Path(__file__).parents[2]/'src/multimodal_jev/rl_update.py'),
                'scope':'controlled HF response fixtures and actual verl loss/LoRA optimizer step; NOT vLLM sampling or RL efficacy'}
        atomic_json(args.output/'result.json',report)
        atomic_json(args.output/'terminal.json',{'exit_code':0,'finished_unix':time.time()})
        print(json.dumps(report,indent=2))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()});raise


if __name__=='__main__':main()

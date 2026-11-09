#!/usr/bin/env python3
"""Real HF actor forward through a padded verl audio batch vs single examples."""
import argparse
import json
from pathlib import Path
import time
import torch
from verl.utils.model import extract_multi_modal_inputs
from multimodal_jev.asr import load_runtime
from multimodal_jev.audio_policy import load_audio,answer_ids,hf_prompt,response_log_probs
from multimodal_jev.task_schema import JevExample,DecisionSchema
from multimodal_jev.verl_audio import make_batch


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--audio',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(4);start=time.monotonic()
    processor,model=load_runtime(args.model,'cpu','float32')
    audio=load_audio(args.audio)
    ex=JevExample('a','train',args.audio,DecisionSchema('s','noul','Language change requested?'),True)
    second=JevExample('b','train',args.audio,DecisionSchema('t','noul','Does this speaker ask to turn off all the lights in the house?'),False)
    examples=[ex,ex,second,second];audios=[audio,audio,audio[:len(audio)//2],audio[:len(audio)//2]]
    responses=[answer_ids(processor.tokenizer,v) for v in (True,False,False,None)]
    batch=make_batch(processor,examples,audios,responses,[[-2.0]*len(x) for x in responses],[1,0,1,0],2)
    b=batch.batch;mm=extract_multi_modal_inputs(batch.non_tensor_batch['multi_modal_inputs'])
    with torch.inference_mode():
        batched=model(input_ids=b['input_ids'],attention_mask=b['attention_mask'],position_ids=b['position_ids'],
                      **mm,use_cache=False).logits
        batch_logp=response_log_probs(batched,b['input_ids'],b['prompts'].shape[-1])
        differences=[]
        for i,(example,wave,ids) in enumerate(zip(examples,audios,responses)):
            one=dict(hf_prompt(processor,example,wave));boundary=one['input_ids'].shape[-1]
            one['input_ids']=torch.cat((one['input_ids'],torch.tensor([ids])),dim=-1)
            one['attention_mask']=torch.ones_like(one['input_ids'])
            one['position_ids']=torch.arange(one['input_ids'].shape[-1]).unsqueeze(0)
            logits=model(**one,use_cache=False).logits
            logp=response_log_probs(logits,one['input_ids'],boundary)
            actual=batch_logp[i,:len(ids)]
            differences.append(float((actual-logp[0]).abs().max()))
            torch.testing.assert_close(actual,logp[0],atol=3e-4,rtol=1e-5)
    receipt={'status':'passed','batch_size':4,'max_token_logprob_abs_error':max(differences),
             'per_example_error':differences,'atol':3e-4,'rtol':1e-5,'elapsed_seconds':time.monotonic()-start,
             'scope':'real HF actor through verl audio collation; padded vs unpadded parity. Not vLLM engine parity.'}
    with args.output.open('x') as handle:json.dump(receipt,handle,indent=2)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

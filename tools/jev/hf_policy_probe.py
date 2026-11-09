#!/usr/bin/env python3
"""Pre-SFT instruction-following diagnostic using genuine HF samples on CPU.

This does not replace the required vLLM rollout backend for RL experiments.
"""
import argparse,json,time
from pathlib import Path
import torch
from multimodal_jev.asr import load_runtime
from multimodal_jev.audio_policy import load_audio,hf_prompt
from multimodal_jev.task_schema import JevExample
from multimodal_jev.reward import compute_score
from multimodal_jev.baseline import file_hash,atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();config=json.loads(args.config.read_text());torch.set_num_threads(4);torch.manual_seed(17)
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'result.json').exists():raise FileExistsError('completed probe')
    start=time.monotonic()
    try:
        processor,model=load_runtime(config['checkpoint'],'cpu','float32')
        wanted=set(config['example_ids']);examples=[]
        with Path(config['manifest']).open() as handle:
            for line in handle:
                row=json.loads(line)
                if row['example_id'] in wanted:examples.append(JevExample.from_record(row))
        assert {ex.example_id for ex in examples}==wanted
        results=[]
        for ex in examples:
            assert file_hash(ex.audio)==ex.metadata['audio_sha256']
            inputs=hf_prompt(processor,ex,load_audio(ex.audio))
            with torch.inference_mode():
                output=model.generate(**inputs,do_sample=True,temperature=1.0,top_p=1.0,top_k=0,
                                      num_return_sequences=2,max_new_tokens=32,return_dict_in_generate=True,output_scores=True)
                logps=model.compute_transition_scores(output.sequences,output.scores,normalize_logits=True)
            for index,sequence in enumerate(output.sequences):
                tokens=sequence[inputs['input_ids'].shape[-1]:].tolist()
                # include first EOS, drop only batch-padding after it
                end=processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
                if end in tokens:tokens=tokens[:tokens.index(end)+1]
                text=processor.tokenizer.decode(tokens,skip_special_tokens=True)
                score=compute_score('fsc',text,json.dumps(ex.to_record()))
                results.append({'example_id':ex.example_id,'replicate':index,'token_ids':tokens,
                                'log_probs':logps[index,:len(tokens)].tolist(),'text':text,'reward':score})
            atomic_json(args.output/'progress.json',{'prompts_done':len(results)//2,'heartbeat_unix':time.time()})
        with (args.output/'samples.jsonl').open('w') as f:
            for row in results:f.write(json.dumps(row)+'\n')
        report={'status':'passed','backend':'HF CPU diagnostic','prompts':len(examples),'samples':len(results),
                'rewards':[r['reward'] for r in results],
                'valid_json_object_outputs':sum(is_object(r['text']) for r in results),
                'elapsed_seconds':time.monotonic()-start,'config':config,'manifest_sha256':file_hash(config['manifest']),
                'scope':'pretrained policy instruction-following diagnosis; no optimization or vLLM result'}
        atomic_json(args.output/'result.json',report)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',{'exit_code':1,'error':repr(exc)});raise
    atomic_json(args.output/'terminal.json',{'exit_code':0})
    print(json.dumps(report,indent=2))


def is_object(text):
    try:return isinstance(json.loads(text),dict)
    except ValueError:return False


if __name__=='__main__':main()

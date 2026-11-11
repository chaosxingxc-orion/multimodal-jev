#!/usr/bin/env python3
"""Direct audio diagnostics with label-free requests and separate offline scoring."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time
from types import SimpleNamespace

from multimodal_jev.baseline import atomic_json, file_hash


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    try:
        import torch
        import torch_npu
        import numpy as np
        from peft import PeftModel
        from multimodal_jev.asr import load_runtime
        from multimodal_jev.audio_policy import hf_prompt,load_audio
        from multimodal_jev.task_schema import DecisionSchema
        from multimodal_jev.thinking import ThinkingSpec
        config=json.loads(args.config.read_text())
        for path,digest in config['input_sha256'].items():
            if file_hash(path)!=digest:raise ValueError('input changed: '+path)
        nodes=sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))
        if nodes!=['davinci4'] or torch.npu.device_count()!=1:
            raise ValueError('physical NPU 4 isolation required')
        torch.set_num_threads(4);torch.manual_seed(17);torch.npu.set_device(0);torch.npu.manual_seed_all(17)
        processor,base=load_runtime(config['model_path'],'npu:0','bfloat16')
        model=PeftModel.from_pretrained(base,config['adapter'],is_trainable=False).eval()
        requests=[json.loads(line) for line in Path(config['requests']).open()]
        predictions={}
        for arm in ('base','sft','sft_silence'):
            records=[]
            with (model.disable_adapter() if arm=='base' else nullcontext()):
                for request in requests:
                    if set(request)!={'id','input','audio_sha256'} or set(request['input'])!={'audio','schema','thinking'}:
                        raise ValueError('label-free requests required')
                    inputs=request['input'];example=SimpleNamespace(
                        audio=inputs['audio'],schema=DecisionSchema.from_dict(inputs['schema']),
                        thinking=ThinkingSpec.from_dict(inputs['thinking']))
                    if file_hash(example.audio)!=request['audio_sha256']:raise ValueError('audio changed')
                    wave=load_audio(example.audio)
                    if arm=='sft_silence':wave=np.zeros_like(wave)
                    batch=hf_prompt(processor,example,wave)
                    batch={k:v.to(device=model.device,dtype=model.dtype if v.is_floating_point() else v.dtype) for k,v in batch.items()}
                    begin=time.monotonic()
                    with torch.inference_mode():
                        output=model.generate(**batch,max_new_tokens=config['max_new_tokens'],do_sample=False)
                    token_ids=output[0,batch['input_ids'].shape[-1]:].tolist()
                    records.append(dict(example_id=request['id'],raw_text=processor.tokenizer.decode(token_ids,skip_special_tokens=True),
                        token_ids=token_ids,seconds=time.monotonic()-begin,reached_limit=len(token_ids)>=config['max_new_tokens']))
                    atomic_json(args.output/'progress.json',dict(arm=arm,done=len(records),total=len(requests),heartbeat_unix=time.time()))
            path=args.output/(arm+'.jsonl');path.write_text(''.join(json.dumps(r)+'\n' for r in records));predictions[arm]=str(path)
        # No target is loaded until all three arms have completed inference.
        from multimodal_jev.temporal_diagnostic import score
        targets=[json.loads(line) for line in Path(config['targets']).open()]
        metrics={arm:score(targets,[json.loads(l) for l in Path(path).open()]) for arm,path in predictions.items()}
        atomic_json(args.output/'result.json',dict(status='passed',metrics=metrics,
            config_sha256=file_hash(args.config),prediction_sha256={k:file_hash(v) for k,v in predictions.items()},
            device_nodes=nodes,elapsed_seconds=time.monotonic()-started,
            scope='Training-population diagnostic, HF eager generation; silence is an input perturbation, not a complete causal audio-grounding test. No held-out or RL claim.'))
        atomic_json(args.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise


if __name__=='__main__':main()

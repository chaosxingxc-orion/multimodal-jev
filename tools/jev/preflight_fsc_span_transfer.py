#!/usr/bin/env python3
"""CPU processor capacity check, using synthetic silence and no FSC labels."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.fsc_span_transfer import compile_request


def run(task,model):
    import numpy as np
    import torch
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.span_contract import encode_span_request,span_labels
    from multimodal_jev.span_request_identity import tensor_identity
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU only')
    torch.set_num_threads(2);processor=Qwen3ASRProcessor.from_pretrained(model,local_files_only=True)
    config=json.loads(Path(task).read_text());rows=[]
    # Longer synthetic clips test capacity without inspecting target utterances.
    for seconds in (1,10,30):
        audio=np.zeros(16000*seconds,dtype=np.float32)
        for variant in ('native','permutation','rebind'):
            features=[]
            for template in ('direct','structured'):
                request,_=compile_request('synthetic-silence.wav',config,template,variant=variant,seed=317)
                batch,positions=encode_span_request(processor,request,audio)
                mapping=span_labels(processor.tokenizer,request.questions)
                length=batch['input_ids'].shape[-1]
                if length>4096:raise ValueError('complete FSC template exceeds context')
                feature=tensor_identity(batch['input_features']);features.append(feature)
                rows.append(dict(seconds=seconds,variant=variant,template=template,tokens=length,
                    positions=positions.tolist(),candidate_widths=[len(m['token_ids']) for m in mapping],
                    input_features=feature))
            if features[0]!=features[1]:raise ValueError('template changed acoustic features')
    return dict(passed=True,requests=len(rows),maximum_tokens=max(r['tokens'] for r in rows),records=rows,
        task_sha256=file_hash(task),model=str(model),scope='CPU processor/token-support capacity on synthetic silence only; no real FSC audio/backend parity or transfer gain.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'result.json',run(a.task,a.model));atomic_json(a.output/'terminal.json',{'exit_code':0})
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise

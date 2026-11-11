#!/usr/bin/env python3
"""Independent HF probabilities for an immutable BF16 adapter and input set.

The existing identity manifest supplies audio/questions and authenticates
preprocessing and tokenization. Old labels and old probabilities never enter
the model; this makes a new reference for cross-backend engineering checks.
"""
import argparse
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json, file_hash


def run(config, output):
    import torch
    import torch_npu
    from peft import PeftModel
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.bf16_lora import cast_lora_bf16
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.span_contract import SpanRequest, SpanQuestion, encode_span_request, span_labels
    from multimodal_jev.task_schema import DecisionSchema
    from multimodal_jev.span_model import NativeAudioSpanModel
    from multimodal_jev.span_runtime_profile import configure_actor_profile
    from multimodal_jev.span_request_identity import bind_span_request
    from multimodal_jev.policy_identity import checkpoint_identity
    from safetensors import safe_open

    if sorted(p.name for p in Path('/dev').glob('davinci[0-9]*')) != ['davinci4']:
        raise ValueError('physical NPU 4 isolation required')
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:
            raise ValueError('pinned input changed: '+path)
    torch.set_num_threads(4);torch.npu.set_device(0);torch.manual_seed(17)
    atomic_json(output/'runtime-profile.json',configure_actor_profile(config['runtime_profile']))
    adapter=Path(config['adapter'])
    with safe_open(str(adapter/'adapter_model.safetensors'),framework='pt',device='cpu') as saved:
        if any(saved.get_tensor(k).dtype!=torch.bfloat16 for k in saved.keys()):
            raise ValueError('BF16 adapter storage required')
    base_id=checkpoint_identity(config['model'])[0];adapter_id=checkpoint_identity(adapter)[0]
    processor,base=load_runtime(config['model'],'npu:0','bfloat16')
    adapted=PeftModel.from_pretrained(base,adapter,is_trainable=False,autocast_adapter_dtype=False).eval()
    cast_lora_bf16(adapted)
    model=NativeAudioSpanModel(adapted.get_base_model()).eval()
    items=json.loads(Path(config['identity']).read_text())['bundles']
    records=[]
    with torch.inference_mode():
        for index,item in enumerate(items):
            raw=item['request']
            request=SpanRequest(raw['audio'],tuple(SpanQuestion(q['id'],DecisionSchema.from_dict(q['schema']))
                for q in raw['questions']),raw['final_id'],tuple(raw['requirements']))
            if file_hash(request.audio)!=item['audio_sha256']:
                raise ValueError('reference audio identity changed')
            audio,receipt=load_audio(request.audio,return_receipt=True)
            batch,positions=encode_span_request(processor,request,audio)
            if (receipt!=item['audio_processing'] or batch['input_ids'].tolist()!=item['input_ids']
                    or positions.tolist()!=item['positions']):
                raise ValueError('reference audio preprocessing/tokenization changed')
            mapping=span_labels(processor.tokenizer,request.questions)
            candidates=[m['token_ids'] for m in mapping]
            request_id,payload=bind_span_request(request,receipt,batch,positions,candidates,base_id+':'+adapter_id)
            inputs={k:v.to('npu:0',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                    for k,v in batch.items()}
            _,logp=model(inputs,positions.to('npu:0'),candidates)
            records.append(dict(id=item['id'],request_id=request_id,request_identity=payload,
                                candidate_logprobs=[p.cpu().tolist() for p in logp]))
            atomic_json(output/'progress.json',dict(completed=index+1,total=len(items),heartbeat_unix=time.time()))
    return dict(records=records,base_id=base_id,adapter_id=adapter_id,runtime_profile=config['runtime_profile'],
                scope='Independent immutable BF16 HF reference; no optimization or task-gain claim')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:
        atomic_json(args.output/'result.json',run(json.loads(args.config.read_text()),args.output))
        atomic_json(args.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise

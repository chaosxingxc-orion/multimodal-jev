#!/usr/bin/env python3
"""Compare actual vLLM long-audio processor with raw and hop-padded HF inputs."""
import argparse
import inspect
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.sfc_audio import load_sfc_audio


def run(args):
    import numpy as np
    import torch
    from transformers import Qwen3OmniMoeProcessor
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.processing.inputs import ProcessorInputs
    from vllm.multimodal.processing.context import TimingContext
    torch.set_num_threads(2)
    if file_hash(args.identity)!=args.identity_hash or file_hash(args.prompt)!=args.prompt_hash:
        raise ValueError('registered prior inputs changed')
    prior=json.loads(args.identity.read_text())
    for name,pin in prior['processor_assets'].items():
        if file_hash(args.model/name)!=pin:raise ValueError('processor assets changed')
    hf=Qwen3OmniMoeProcessor.from_pretrained(args.model,local_files_only=True)
    cfg=ModelConfig(model=str(args.model),tokenizer=str(args.model),dtype='bfloat16',
        max_model_len=32768,enforce_eager=True,limit_mm_per_prompt={'audio':7,'image':0,'video':0})
    processor=MULTIMODAL_REGISTRY.create_processor(cfg,tokenizer=hf.tokenizer)
    saved=args.prompt.read_text();start='<|im_start|>system\n';end='<|im_end|>\n'
    if not saved.startswith(start):raise ValueError('system layout changed')
    system=saved[len(start):].split(end,1)[0]
    atomic_json(args.output/'identity.json',dict(prior_identity_sha256=args.identity_hash,
        prompt_sha256=args.prompt_hash,model=str(args.model),context_limit=32768,audio_limit=7,
        actual_processor_type=type(processor).__name__,
        vllm_processor_source_sha256=file_hash(inspect.getfile(type(processor))),
        hf_processor_source_sha256=file_hash(inspect.getfile(type(hf))),
        model_loaded=False,answers_read=False,source_sha256=file_hash(__file__)))
    records=[]
    for name,audios in prior['selections']:
        begin=time.monotonic();waves=[];content=[]
        atomic_json(args.output/'progress.json',dict(selection=name,phase='audio',heartbeat_unix=time.time()))
        for a in audios:
            path=args.root/a['path']
            if file_hash(path)!=a['sha256']:raise ValueError('waveform identity changed')
            waveform,_=load_sfc_audio(path);waves.append(waveform)
            if 'turn' in a:content.append(dict(type='text',text=f"Conversation turn {a['turn']}, speaker role {a['role']}:"))
            content.append(dict(type='audio',audio=waveform))
        prompt=hf.apply_chat_template([dict(role='system',content=system),dict(role='user',content=content)],
                                      tokenize=False,add_generation_prompt=True)
        atomic_json(args.output/'progress.json',dict(selection=name,phase='vllm_processor',heartbeat_unix=time.time()))
        parsed=processor.info.get_data_parser().parse_mm_data({'audio':[(w,16000) for w in waves]})
        processed=processor.apply(ProcessorInputs(prompt=hf.tokenizer.encode(prompt,add_special_tokens=False),
                                                  mm_data_items=parsed),TimingContext(enabled=False))
        data=processed['mm_kwargs'].get_data();feature=data['input_audio_features']
        frames=data['audio_feature_lengths'].tolist()
        if feature.ndim!=2 or feature.shape[1]!=sum(frames) or not torch.isfinite(feature).all():
            raise ValueError('complete concatenated finite vLLM features required')
        hop=hf.feature_extractor.hop_length
        padded=[np.pad(w,(0,(-len(w))%hop)) for w in waves]
        comparisons={}
        for mode,inputs in [('raw',waves),('hop_padded',padded)]:
            batch=hf(text=[prompt],audio=inputs,sampling_rate=16000,return_tensors='pt',padding=True,truncation=False)
            mask=batch['feature_attention_mask'];observed=mask.sum(-1).tolist()
            reference=batch['input_features'].permute(0,2,1)[mask.bool()].permute(1,0)
            shape_equal=list(feature.shape)==list(reference.shape)
            # vLLM processing context casts floating inputs to model dtype.
            # Compare at that boundary without relaxing feature tolerance;
            # retain the unquantized FP32 difference as separate evidence.
            cast_reference=reference.to(feature.dtype)
            comparisons[mode]=dict(frames=observed,shape=list(reference.shape),
                vllm_dtype=str(feature.dtype),hf_dtype=str(reference.dtype),
                token_ids_equal=processed['prompt_token_ids']==batch['input_ids'][0].tolist(),
                shape_equal=shape_equal,
                unquantized_fp32_max_error=float((feature.float()-reference.float()).abs().max()) if shape_equal else None,
                max_feature_error=float((feature.float()-cast_reference.float()).abs().max()) if shape_equal else None,
                features_close=bool(torch.allclose(feature.float(),cast_reference.float(),atol=1e-5,rtol=1e-5)) if shape_equal else False)
            del batch,mask,reference,cast_reference
        expected=[len(w)//hop for w in padded]
        record=dict(selection=name,original_samples=list(map(len,waves)),hop_padded_samples=list(map(len,padded)),
            vllm_frames=frames,expected_untruncated_frames=expected,
            vllm_shape=list(feature.shape),input_tokens=len(processed['prompt_token_ids']),
            comparisons=comparisons,seconds=time.monotonic()-begin,
            fits_32768_with_1024_output=len(processed['prompt_token_ids'])+1024<=32768)
        atomic_json(args.output/(name+'.json'),record);records.append(record)
        if frames!=expected:raise ValueError('vLLM truncated frame coverage')
        match=comparisons['hop_padded']
        if not match['token_ids_equal'] or not match['features_close']:
            raise ValueError('hop-aligned HF/vLLM processor parity failed')
        if not record['fits_32768_with_1024_output']:raise ValueError('full context overflow')
        del feature,processed,data,waves,padded
    atomic_json(args.output/'result.json',dict(status='passed',records=records,
        model_loaded=False,answers_read=False,scope='Real vLLM and HF CPU processors for two header-selected extremes; 32768 context with 7-audio configuration. No engine, weight loading, NPU memory, full-dataset fit or task-quality claim.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','model','identity','prompt','output'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('identity','prompt'):p.add_argument('--'+key+'-hash',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:run(a)
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0))

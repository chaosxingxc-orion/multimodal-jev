#!/usr/bin/env python3
"""Run vLLM's real multimodal processor on CPU and compare HF actor inputs."""
import argparse
import json
from pathlib import Path
import torch
from transformers import Qwen3ASRProcessor
from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.multimodal.processing.context import TimingContext
from multimodal_jev.audio_policy import load_audio,hf_prompt,vllm_request
from multimodal_jev.task_schema import JevExample,DecisionSchema


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--audio',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(4)
    hf=Qwen3ASRProcessor.from_pretrained(args.model,local_files_only=True)
    cfg=ModelConfig(model=args.model,tokenizer=args.model,dtype='float32',max_model_len=2048,enforce_eager=True)
    processor=MULTIMODAL_REGISTRY.create_processor(cfg,tokenizer=hf.tokenizer)
    audio=load_audio(args.audio)
    ex=JevExample('probe','train',args.audio,DecisionSchema('s','noul','Language change requested?'),True)
    results=[]
    for waveform in (audio,audio[:len(audio)//2]):
        request=vllm_request(hf,ex,waveform)
        raw_ids=hf.tokenizer.encode(request['prompt'],add_special_tokens=False)
        items=processor.info.get_data_parser().parse_mm_data(request['multi_modal_data'])
        processed=processor.apply(ProcessorInputs(prompt=raw_ids,mm_data_items=items),TimingContext(enabled=False))
        actor=hf_prompt(hf,ex,waveform)
        assert processed['prompt_token_ids']==actor['input_ids'][0].tolist()
        data=processed['mm_kwargs'].get_data()
        features=data['input_audio_features'];frames=int(actor['input_features_mask'].sum())
        reference=actor['input_features'][0,:,:frames]
        if features.ndim==3:features=features[0]
        assert features.shape==reference.shape,(features.shape,reference.shape)
        diff=float((features-reference).abs().max())
        torch.testing.assert_close(features,reference,atol=1e-5,rtol=1e-5)
        results.append({'samples':len(waveform),'prompt_tokens':len(processed['prompt_token_ids']),
                        'feature_shape':list(features.shape),'max_feature_error':diff})
    receipt={'status':'passed','results':results,'scope':'actual vLLM multimodal processor and HF actor token/features parity; no engine or device execution'}
    with args.output.open('x') as handle:json.dump(receipt,handle,indent=2)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

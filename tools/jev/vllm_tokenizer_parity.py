#!/usr/bin/env python3
"""Direct vLLM-bundled HF processor parity, without loading model registry."""
import argparse,json
from pathlib import Path
import torch
from transformers import Qwen3ASRProcessor as HFProcessor
from vllm.transformers_utils.processors.qwen3_asr import Qwen3ASRProcessor as RolloutProcessor
from multimodal_jev.audio_policy import load_audio,hf_prompt,vllm_request
from multimodal_jev.task_schema import JevExample,DecisionSchema


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--audio',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(4)
    actor=HFProcessor.from_pretrained(args.model,local_files_only=True)
    rollout=RolloutProcessor.from_pretrained(args.model,local_files_only=True)
    wave=load_audio(args.audio)
    ex=JevExample('probe','train',args.audio,DecisionSchema('s','noul','Language change requested?'),True)
    cases=[]
    for audio in [wave,wave[:len(wave)//2]]:
        raw=vllm_request(actor,ex,audio)
        hf=hf_prompt(actor,ex,audio)
        result=rollout(text=[raw['prompt']],audio=[audio],sampling_rate=16000,return_tensors='pt')
        assert torch.equal(hf['input_ids'],result['input_ids'])
        frame_mask=result['feature_attention_mask']
        print(json.dumps({'hf_feature_shape':list(hf['input_features'].shape),'vllm_feature_shape':list(result['input_features'].shape),'hf_mask_shape':list(hf['input_features_mask'].shape),'vllm_mask_shape':list(frame_mask.shape),'hf_frames':hf['input_features_mask'].sum().item(),'vllm_frames':frame_mask.sum().item()}),flush=True)
        frames=int(hf['input_features_mask'].sum())
        assert frames==int(frame_mask.sum())
        hf_features=hf['input_features'][...,:frames]
        rollout_features=result['input_features'][...,:frames]
        torch.testing.assert_close(hf_features,rollout_features,atol=1e-5,rtol=1e-5)
        cases.append({'samples':len(audio),'tokens':hf['input_ids'].shape[-1],
            'feature_error':float((hf_features-rollout_features).abs().max())})
    receipt={'status':'passed','cases':cases,'scope':'direct vLLM bundled HF processor only; full multimodal registry and NPU engine not exercised'}
    with args.output.open('x') as handle:json.dump(receipt,handle,indent=2)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

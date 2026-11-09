#!/usr/bin/env python3
"""Real HF two-audio forward and direct vLLM processor parity on TRAIN only."""
import argparse
import json
from pathlib import Path
import time
import torch
from multimodal_jev.asr import load_runtime
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.sfc_audio import hf_inputs, render_prompt, vllm_request, load_sfc_audio


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:
        torch.set_num_threads(4);start=time.monotonic()
        with a.manifest.open() as handle:
            rows=[json.loads(line) for line in handle]
        candidates=[r for r in rows if len(r['input']['audios'])==2]
        row=min(candidates,key=lambda r:r['example_id'])
        if row['metadata']['official_split']!='train_id':raise ValueError('TRAIN only')
        inputs=row['input'];waves=[];conversions=[]
        for path,digest in zip(inputs['audios'],row['metadata']['audio_sha256']):
            if file_hash(path)!=digest:raise ValueError('audio changed')
            wave,conversion=load_sfc_audio(path);waves.append(wave);conversions.append(conversion)
        atomic_json(a.output/'progress.json',{'phase':'load','heartbeat_unix':time.time()})
        processor,model=load_runtime(a.model,'cpu','float32')
        batch=hf_inputs(processor,inputs,waves)
        # Reversing waveforms must reverse feature rows, without changing text.
        swapped=hf_inputs(processor,inputs,waves[::-1])
        order_error=float((batch['input_features']-swapped['input_features'].flip(0)).abs().max())
        torch.testing.assert_close(batch['input_features'],swapped['input_features'].flip(0),atol=1e-5,rtol=1e-5)
        assert torch.equal(batch['input_features_mask'],swapped['input_features_mask'].flip(0))
        from vllm.transformers_utils.processors.qwen3_asr import Qwen3ASRProcessor
        rollout=Qwen3ASRProcessor.from_pretrained(a.model,local_files_only=True)
        request=vllm_request(inputs,waves)
        other=rollout(text=[request['prompt']],audio=waves,sampling_rate=16000,return_tensors='pt')
        assert torch.equal(batch['input_ids'],other['input_ids'])
        errors=[]
        for i in range(2):
            frames=int(batch['input_features_mask'][i].sum())
            print(json.dumps({'audio':i,'hf_frames':frames,'vllm_frames':int(other['feature_attention_mask'][i].sum()),
                              'samples':len(waves[i]),'hf_shape':list(batch['input_features'].shape),
                              'vllm_shape':list(other['input_features'].shape)}),flush=True)
            assert frames==int(other['feature_attention_mask'][i].sum())
            left=batch['input_features'][i,:,:frames];right=other['input_features'][i,:,:frames]
            torch.testing.assert_close(left,right,atol=1e-5,rtol=1e-5)
            errors.append(float((left-right).abs().max()))
        atomic_json(a.output/'progress.json',{'phase':'forward','heartbeat_unix':time.time()})
        with torch.inference_mode():
            output=model(**batch,use_cache=False,logits_to_keep=1)
        assert torch.isfinite(output.logits).all()
        report={'status':'passed','example_id':row['example_id'],'audio_count':2,
                'input_shapes':{k:list(v.shape) for k,v in batch.items()},'audio_conversion':conversions,
                'audio_order_verified':True,'audio_order_max_error':order_error,'direct_vllm_processor_token_ids_equal':True,
                'valid_feature_errors':errors,'finite_hf_forward':True,
                'elapsed_seconds':time.monotonic()-start,'manifest_sha256':file_hash(a.manifest),
                'adapter_sha256':file_hash(Path(__file__).parents[2]/'src/multimodal_jev/sfc_audio.py'),
                'scope':'TRAIN-only HF forward and direct bundled processor; no engine execution, generation or SFC accuracy'}
        atomic_json(a.output/'result.json',report)
        atomic_json(a.output/'terminal.json',{'exit_code':0,'finished_unix':time.time()})
        print(json.dumps(report,indent=2))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()})
        raise


if __name__=='__main__':main()

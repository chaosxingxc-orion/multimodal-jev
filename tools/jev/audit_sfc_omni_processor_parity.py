#!/usr/bin/env python3
"""Complete registered TRAIN6/78 decisions: actual HF/vLLM input comparison.

No model or labels. Preserve the raw actor mismatch as evidence; hop-aligned HF
is a separately named diagnostic, not a mutation of running baseline inputs.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.omni_actor_batch import prepare, evidence
from multimodal_jev.sfc_audio import load_sfc_audio
from multimodal_jev.sfc_jev_runtime import messages
from sample_sfc_omni_learning import load_inputs


def run(args):
    import numpy as np
    import torch
    from transformers import Qwen3OmniMoeProcessor
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.processing.inputs import ProcessorInputs
    from vllm.multimodal.processing.context import TimingContext
    torch.set_num_threads(2); start=time.monotonic()
    if file_hash(args.config)!=args.config_hash:raise ValueError('sampling config changed')
    cfg=json.loads(args.config.read_text());base,rows,preflight,checks=load_inputs(cfg)
    if {k:importlib.metadata.version(k) for k in base['versions']}!=base['versions']:
        raise ValueError('fixed runtime required')
    for name,pin in preflight['processor_assets'].items():
        if file_hash(Path(base['audio_model'])/name)!=pin:raise ValueError('processor asset changed')
    hf=Qwen3OmniMoeProcessor.from_pretrained(base['audio_model'],local_files_only=True)
    raw=prepare(hf,rows)
    for key,values in evidence(raw).items():
        if any(checks[key][name]!=value for name,value in values.items()):
            raise ValueError('prior actor input identity changed')
    mc=ModelConfig(model=base['audio_model'],tokenizer=base['audio_model'],dtype='bfloat16',
        max_model_len=4096,enforce_eager=True,limit_mm_per_prompt={'audio':2,'image':0,'video':0})
    processor=MULTIMODAL_REGISTRY.create_processor(mc,tokenizer=hf.tokenizer)
    atomic_json(args.output/'identity.json',dict(config=cfg,config_sha256=args.config_hash,
        selected_ids=[r['id'] for r in rows],decisions=sum(len(r['input']['decisions']) for r in rows),
        vllm_processor_sha256=file_hash(inspect.getfile(type(processor))),
        hf_processor_sha256=file_hash(inspect.getfile(type(hf))),
        runner_sha256=file_hash(__file__),model_loaded=False,targets_read=False,
        feature_atol=1e-5,feature_rtol=1e-5,model_dtype='bfloat16'))
    records=[]; hop=hf.feature_extractor.hop_length
    with (args.output/'comparisons.jsonl').open('x') as handle:
        for row in rows:
            waves=[]
            for path,pin in zip(row['input']['context']['audios'],row['audio_sha256'],strict=True):
                if file_hash(path)!=pin:raise ValueError('audio changed')
                waves.append(load_sfc_audio(path)[0])
            padded=[np.pad(w,(0,(-len(w))%hop)) for w in waves]
            for decision in row['input']['decisions']:
                key=(row['id'],decision['id']);old=raw[key]
                prompt=hf.apply_chat_template(messages(row,decision['id'],waves),tokenize=False,add_generation_prompt=True)
                if hashlib.sha256(prompt.encode()).hexdigest()!=old['prompt_sha256']:
                    raise ValueError('raw renderer changed')
                parsed=processor.info.get_data_parser().parse_mm_data({'audio':[(w,16000) for w in waves]})
                processed=processor.apply(ProcessorInputs(prompt=hf.tokenizer.encode(prompt,add_special_tokens=False),
                    mm_data_items=parsed),TimingContext(enabled=False))
                data=processed['mm_kwargs'].get_data();feature=data['input_audio_features']
                frame_lengths=data['audio_feature_lengths'].tolist();expected=[len(w)//hop for w in padded]
                aligned=hf(text=[prompt],audio=padded,sampling_rate=16000,return_tensors='pt',padding=True,truncation=False)
                comparisons={}
                for name,inputs,ids in [('raw_actor',old,old['input_ids'].tolist()),
                                       ('hop_aligned_actor',aligned,aligned['input_ids'][0].tolist())]:
                    mask=inputs['feature_attention_mask'];ref=inputs['input_features'].permute(0,2,1)[mask.bool()].permute(1,0)
                    same_shape=feature.shape==ref.shape
                    delta=(feature.float()-ref.to(feature.dtype).float()).abs() if same_shape else None
                    comparisons[name]=dict(token_ids_equal=ids==processed['prompt_token_ids'],
                        frames=mask.sum(-1).tolist(),shape_equal=same_shape,
                        max_feature_error=float(delta.max()) if delta is not None else None,
                        features_close=bool(torch.allclose(feature.float(),ref.to(feature.dtype).float(),
                            atol=1e-5,rtol=1e-5)) if same_shape else False)
                record=dict(example_id=row['id'],decision_id=decision['id'],audio_count=len(waves),
                    original_samples=list(map(len,waves)),hop_padded_samples=list(map(len,padded)),
                    vllm_frames=frame_lengths,expected_frames=expected,
                    vllm_input_tokens=len(processed['prompt_token_ids']),comparisons=comparisons)
                handle.write(json.dumps(record)+'\n');handle.flush();records.append(record)
                atomic_json(args.output/'progress.json',dict(done=len(records),total=78,heartbeat_unix=time.time()))
    if len(rows)!=6 or len(records)!=78:raise ValueError('complete registered roster required')
    count=lambda mode,field:sum(not r['comparisons'][mode][field] for r in records)
    aligned_pass=all(r['comparisons']['hop_aligned_actor']['token_ids_equal']
        and r['comparisons']['hop_aligned_actor']['features_close'] and r['vllm_frames']==r['expected_frames']
        and r['vllm_input_tokens']+1024<=4096 for r in records)
    result=dict(status='passed' if aligned_pass else 'failed',rows=len(rows),decisions=len(records),
        raw_actor_token_mismatches=count('raw_actor','token_ids_equal'),
        raw_actor_feature_mismatches=count('raw_actor','features_close'),
        aligned_actor_token_mismatches=count('hop_aligned_actor','token_ids_equal'),
        aligned_actor_feature_mismatches=count('hop_aligned_actor','features_close'),
        max_aligned_feature_error=max(r['comparisons']['hop_aligned_actor']['max_feature_error'] or 0 for r in records),
        comparisons_sha256=file_hash(args.output/'comparisons.jsonl'),elapsed_seconds=time.monotonic()-start,
        model_loaded=False,targets_read=False,
        scope='Actual complete TRAIN6/78 CPU HF and vLLM processors at model dtype. Raw existing actor and explicit hop-aligned diagnostic separately reported; no weights, logits, sampling, policy updates or input mutation.')
    atomic_json(args.output/'result.json',result)
    if not aligned_pass:raise ValueError('hop-aligned processor parity failed')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('config','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--config-hash',required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:run(args)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(args.output/'terminal.json',dict(exit_code=0))

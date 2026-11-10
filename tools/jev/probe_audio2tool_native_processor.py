#!/usr/bin/env python3
"""Actual HF/vLLM processing of fixed native requests before NPU dispatch."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

from multimodal_jev.audio2tool_audio import prepare_request
from multimodal_jev.audio2tool_input import validate
from multimodal_jev.baseline import atomic_json, file_hash
from check_audio2tool_audio_loader import INPUT_SHA, BUDGET_SHA


def selections(rows, budgets):
    if (len(rows)!=30733 or len({r['id'] for r in rows})!=30733
            or [r['id'] for r in rows]!=[b['id'] for b in budgets]):
        raise ValueError('complete ordered release roster required')
    chosen={}
    for row,budget in zip(rows,budgets,strict=True):
        validate(row['input'])
        if (row['split']!='external_test' or budget['mode']!=row['input']['mode']
                or len(budget['frames'])!=len(row['input']['audios'])
                or budget['input_tokens']+1024>32768):
            raise ValueError('registered complete input and context required')
        mode=row['input']['mode']
        if mode not in chosen or budget['input_tokens']>chosen[mode][1]['input_tokens']:
            chosen[mode]=(row,budget)
    # Include the largest number of audio clips, not just maximum token length.
    index=max(range(len(rows)),key=lambda i:(len(rows[i]['input']['audios']),budgets[i]['input_tokens']))
    if len(rows[index]['input']['audios'])!=7:raise ValueError('registered seven-audio maximum required')
    chosen['maximum_audio_count']=(rows[index],budgets[index])
    if set(chosen)!={'single_audio','dialogue_last_user','foreground_mixture','maximum_audio_count'}:
        raise ValueError('all native input modes required')
    return chosen


def run(args):
    import torch
    from transformers import Qwen3OmniMoeProcessor
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.processing.inputs import ProcessorInputs
    from vllm.multimodal.processing.context import TimingContext
    torch.set_num_threads(2);start=time.monotonic()
    if file_hash(args.inputs)!=INPUT_SHA or file_hash(args.budgets)!=BUDGET_SHA:
        raise ValueError('registered native inputs and budgets changed')
    if file_hash(args.prior_identity)!='69323fb033f9e93ea46ad154605f8dcd2523c8d5d6a80fdfadd6e2cc0661e199':
        raise ValueError('registered processor asset identity changed')
    assets=json.loads(args.prior_identity.read_text())['processor_assets']
    for name,sha in assets.items():
        if file_hash(args.model/name)!=sha:raise ValueError('processor assets changed')
    rows=[json.loads(l) for l in args.inputs.open()]
    budgets=[json.loads(l) for l in args.budgets.open()]
    chosen=selections(rows,budgets);registry=args.registry.read_bytes()
    hf=Qwen3OmniMoeProcessor.from_pretrained(args.model,local_files_only=True)
    config=ModelConfig(model=str(args.model),tokenizer=str(args.model),dtype='bfloat16',
        max_model_len=32768,enforce_eager=True,limit_mm_per_prompt={'audio':7,'image':0,'video':0})
    processor=MULTIMODAL_REGISTRY.create_processor(config,tokenizer=hf.tokenizer)
    atomic_json(args.output/'identity.json',dict(inputs_sha256=INPUT_SHA,budgets_sha256=BUDGET_SHA,
        processor_assets=assets,selected={k:r['id'] for k,(r,b) in chosen.items()},
        hf_processor_source_sha256=file_hash(inspect.getfile(type(hf))),
        vllm_processor_source_sha256=file_hash(inspect.getfile(type(processor))),
        registry_sha256=file_hash(args.registry),runner_sha256=file_hash(__file__),
        model_loaded=False,annotations_read=False))
    results=[]
    for name,(row,budget) in sorted(chosen.items()):
        begin=time.monotonic()
        atomic_json(args.output/'progress.json',dict(phase='processing',selection=name,heartbeat_unix=time.time()))
        request,receipts=prepare_request(hf,row['input'],args.root,registry)
        waves=[w for w,rate in request['multi_modal_data']['audio']]
        parsed=processor.info.get_data_parser().parse_mm_data(request['multi_modal_data'])
        processed=processor.apply(ProcessorInputs(
            prompt=hf.tokenizer.encode(request['prompt'],add_special_tokens=False),
            mm_data_items=parsed),TimingContext(enabled=False))
        data=processed['mm_kwargs'].get_data()
        feature=data['input_audio_features'];frames=data['audio_feature_lengths'].tolist()
        batch=hf(text=[request['prompt']],audio=waves,sampling_rate=16000,
                 return_tensors='pt',padding=True,truncation=False)
        mask=batch['feature_attention_mask']
        reference=batch['input_features'].permute(0,2,1)[mask.bool()].permute(1,0).to(feature.dtype)
        token_equal=processed['prompt_token_ids']==batch['input_ids'][0].tolist()
        expected=[r['expected_frames'] for r in receipts]
        if (frames!=expected or expected!=budget['frames'] or mask.sum(-1).tolist()!=expected
                or feature.dtype!=torch.bfloat16 or feature.ndim!=2 or feature.shape[1]!=sum(expected)
                or feature.shape!=reference.shape or not torch.isfinite(feature).all()
                or not torch.isfinite(reference).all() or not token_equal
                or len(processed['prompt_token_ids'])!=budget['input_tokens']):
            raise ValueError('full frames, finite features or exact input tokens differ')
        delta=float((feature.float()-reference.float()).abs().max())
        record=dict(selection=name,id=row['id'],mode=row['input']['mode'],audio_receipts=receipts,
            frames=frames,feature_shape=list(feature.shape),input_tokens=len(processed['prompt_token_ids']),
            prompt_sha256=hashlib.sha256(request['prompt'].encode()).hexdigest(),
            input_ids_sha256=hashlib.sha256(json.dumps(processed['prompt_token_ids']).encode()).hexdigest(),
            input_ids_equal=token_equal,bf16_features_equal=bool(torch.equal(feature,reference)),
            bf16_feature_max_error=delta,seconds=time.monotonic()-begin)
        atomic_json(args.output/(name+'.json'),record)
        if not record['bf16_features_equal']:raise ValueError('exact BF16 feature parity failed')
        results.append(record)
        del waves,request,processed,data,feature,batch,mask,reference
    atomic_json(args.output/'result.json',dict(status='passed',records=results,
        identity_sha256=file_hash(args.output/'identity.json'),elapsed_seconds=time.monotonic()-start,
        model_loaded=False,annotations_read=False,
        scope='Four deterministic native-input cases: three mode token maxima and largest seven-clip input. Real CPU HF/vLLM processors with exact BF16 features and token IDs. No engine, model generation, NPU capacity or task performance acceptance.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','model','inputs','budgets','registry','prior-identity','output'):
        p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:run(args)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(args.output/'terminal.json',dict(exit_code=0))

#!/usr/bin/env python3
"""All Jev context budgets and actual HF/vLLM processing at typed boundaries."""
import argparse
from bisect import bisect_right
from collections import Counter,defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import time

from multimodal_jev.audio2tool_audio import load_audio
from multimodal_jev.audio2tool_engine import prior_records
from multimodal_jev.audio2tool_inference import read_population,digest,write_once
from multimodal_jev.audio2tool_jev_runtime import messages
from multimodal_jev.baseline import atomic_json,file_hash

MANIFEST_SHA='56921792b06636d83bd6b8a21e49ae0f887f3de68a9813d75fe1b98d83c0f062'


def run(args):
    import torch
    from transformers import Qwen3OmniMoeProcessor
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.processing.inputs import ProcessorInputs
    from vllm.multimodal.processing.context import TimingContext
    torch.set_num_threads(2);start=time.monotonic()
    if file_hash(args.release/'MANIFEST.json')!=MANIFEST_SHA:raise ValueError('registered complete Jev release required')
    manifest=json.loads((args.release/'MANIFEST.json').read_text())
    for file,key in [('external_test.requests.jsonl','requests_sha256'),('task-bank.json','task_bank_file_sha256')]:
        if file_hash(args.release/file)!=manifest[key]:raise ValueError('Jev actor input changed')
    bank=json.loads((args.release/'task-bank.json').read_text());registry=args.registry.read_bytes()
    rows=[json.loads(l) for l in (args.release/'external_test.requests.jsonl').open()]
    native,budgets=read_population(args.native_inputs,args.budgets)
    if len(rows)!=len(native):raise ValueError('complete native/Jev roster required')
    _,prior=prior_records(args.processor_run)
    for name,pin in prior['processor_assets'].items():
        if file_hash(args.model/name)!=pin:raise ValueError('original processor assets changed')
    hf=Qwen3OmniMoeProcessor.from_pretrained(args.model,local_files_only=True)
    if file_hash(inspect.getfile(type(hf)))!=prior['hf_processor_source_sha256']:raise ValueError('fixed HF processor required')
    cfg=ModelConfig(model=str(args.model),tokenizer=str(args.model),dtype='bfloat16',max_model_len=32768,
                    enforce_eager=True,limit_mm_per_prompt={'audio':7,'image':0,'video':0})
    vp=MULTIMODAL_REGISTRY.create_processor(cfg,tokenizer=hf.tokenizer)
    if file_hash(inspect.getfile(type(vp)))!=prior['vllm_processor_source_sha256']:raise ValueError('fixed vLLM processor required')
    layouts=defaultdict(list);audio_token=hf.tokenizer.convert_tokens_to_ids('<|audio_pad|>')
    for index,(row,n,budget) in enumerate(zip(rows,native,budgets,strict=True)):
        if (row['id']!=n['id'] or row['native_request_sha256']!=digest(n) or row['input']['context']!=n['input']
                or row['input']['task_bank_sha256']!=manifest['identity']['task_bank_sha256']):
            raise ValueError('full audio-only native/Jev input binding differs')
        inp=n['input'];layout=(inp['mode'],tuple((a.get('turn'),a.get('role')) for a in inp['audios']))
        layouts[layout].append(index)
    if len(layouts)!=6 or len(bank['decisions'])!=457:raise ValueError('complete layouts and decision bank required')
    maxima={};profiles=[];over={str(n):Counter() for n in (4096,8192,16384,32768)};max_input=0
    for layout_number,(layout,indices) in enumerate(layouts.items()):
        totals=sorted(sum(budgets[i]['audio_tokens']) for i in indices)
        max_index=max(indices,key=lambda i:sum(budgets[i]['audio_tokens']))
        example=rows[max_index];audio_count=len(example['input']['context']['audios'])
        max_audio=max(max(budgets[i]['audio_tokens']) for i in indices)
        for number,decision in enumerate(bank['decisions']):
            if number%100==0:
                atomic_json(args.output/'progress.json',dict(phase='layout_decisions',layout=layout_number,decision=number,heartbeat_unix=time.time()))
            prompt=hf.apply_chat_template(messages(example,decision['id'],['']*audio_count,bank,registry),
                tokenize=False,add_generation_prompt=True)
            ids=hf.tokenizer(prompt,add_special_tokens=False)['input_ids']
            if ids.count(audio_token)!=audio_count:raise ValueError('complete placeholders required')
            text_tokens=len(ids)-audio_count
            # Check special-token expansion at both the minimum and the largest
            # clip length for every layout/decision, not only one prompt kind.
            expanded=prompt.replace('<|audio_pad|>','<|audio_pad|>'*max_audio)
            if len(hf.tokenizer(expanded,add_special_tokens=False)['input_ids'])!=text_tokens+audio_count*max_audio:
                raise ValueError('nonadditive audio token expansion')
            total=text_tokens+totals[-1];max_input=max(max_input,total)
            for limit,counts in over.items():counts[decision['decision_type']]+=len(totals)-bisect_right(totals,int(limit)-1024-text_tokens)
            profile=dict(layout=layout_number,mode=layout[0],roles=layout[1],records=len(indices),decision_id=decision['id'],
                decision_type=decision['decision_type'],text_tokens=text_tokens,max_input_tokens=total,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
            profiles.append(profile)
            key=layout[0]+':'+decision['decision_type']
            if key not in maxima or total>maxima[key]['total']:
                maxima[key]=dict(index=max_index,decision_id=decision['id'],total=total)
            if audio_count==7:
                key='seven_audio:'+decision['decision_type']
                if key not in maxima or total>maxima[key]['total']:
                    maxima[key]=dict(index=max_index,decision_id=decision['id'],total=total)
    with (args.output/'profiles.jsonl').open('x') as handle:
        for profile in profiles:handle.write(json.dumps(profile)+'\n')
    if len(maxima)!=12:raise ValueError('all three decision kinds across three modes plus seven-audio boundary required')
    records=[]
    for name,case in sorted(maxima.items()):
        row=rows[case['index']];budget=budgets[case['index']]
        atomic_json(args.output/'progress.json',dict(phase='real_processors',case=name,heartbeat_unix=time.time()))
        waves,receipts=load_audio(row['input']['context'],args.root)
        prompt=hf.apply_chat_template(messages(row,case['decision_id'],waves,bank,registry),tokenize=False,add_generation_prompt=True)
        parsed=vp.info.get_data_parser().parse_mm_data({'audio':[(w,16000) for w in waves]})
        processed=vp.apply(ProcessorInputs(prompt=hf.tokenizer.encode(prompt,add_special_tokens=False),mm_data_items=parsed),TimingContext(enabled=False))
        data=processed['mm_kwargs'].get_data();feature=data['input_audio_features'];frames=data['audio_feature_lengths'].tolist()
        batch=hf(text=[prompt],audio=waves,sampling_rate=16000,return_tensors='pt',padding=True,truncation=False)
        mask=batch['feature_attention_mask'];reference=batch['input_features'].permute(0,2,1)[mask.bool()].permute(1,0).to(feature.dtype)
        if (frames!=budget['frames'] or frames!=[r['expected_frames'] for r in receipts] or mask.sum(-1).tolist()!=frames
                or processed['prompt_token_ids']!=batch['input_ids'][0].tolist()
                or len(processed['prompt_token_ids'])!=case['total'] or feature.dtype!=torch.bfloat16
                or not torch.isfinite(feature).all() or not torch.equal(feature,reference)):
            raise ValueError('complete frames, exact token IDs or exact BF16 features differ')
        record=dict(case=name,id=row['id'],decision_id=case['decision_id'],input_tokens=case['total'],frames=frames,
            audio_receipts=receipts,input_ids_sha256=digest(processed['prompt_token_ids']),
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),token_ids_exact=True,bf16_features_exact=True)
        write_once(args.output/(name.replace(':','-')+'.json'),record);records.append(record)
        del waves,batch,mask,feature,reference,processed,data
    write_once(args.output/'result.json',dict(status='passed',requests=len(rows),decisions=len(rows)*457,
        layouts=len(layouts),profiles=len(profiles),max_input_tokens=max_input,
        exceeded_with_1024_output={k:dict(v) for k,v in over.items()},real_processor_cases=records,
        profiles_sha256=file_hash(args.output/'profiles.jsonl'),manifest_sha256=MANIFEST_SHA,
        source_sha256={n:file_hash(Path(__file__).resolve().parents[2]/n) for n in (
            'tools/jev/audit_audio2tool_jev_context.py','src/multimodal_jev/audio2tool_jev_runtime.py',
            'src/multimodal_jev/audio2tool_jev.py','src/multimodal_jev/audio2tool_audio.py')},
        elapsed_seconds=time.monotonic()-start,model_loaded=False,targets_read=False,
        scope='Full 30733x457 input budget via all 2742 real-tokenizer layout/decision profiles and exact expansion checks. Twelve real HF/vLLM audio processor cases. No NPU engine, inference score, target access or training.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','release','native-inputs','budgets','model','registry','processor-run','output'):p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:run(args)
    except BaseException as exc:
        write_once(args.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    write_once(args.output/'terminal.json',dict(exit_code=0))

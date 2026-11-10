#!/usr/bin/env python3
"""Full native-input budgets plus actual HF processor on each mode's maximum.

Uses pinned answer-free inputs and the real production message renderer.
No model weights, annotations, engine or inference dispatch.
"""
import argparse
from collections import Counter
import inspect
import json
from pathlib import Path
import time

from multimodal_jev.audio2tool_input import messages, validate
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.omni_waveform_boundary import align_to_hop
from multimodal_jev.sfc_audio import load_sfc_audio
from audit_audio2tool_context_population import samples_and_frames


def run(args):
    import torch
    from transformers import Qwen3OmniMoeProcessor
    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import _get_feat_extract_output_lengths
    torch.set_num_threads(2);begin=time.monotonic()
    for k in ('inputs','headers','registry'):
        if file_hash(getattr(args,k))!=getattr(args,k+'_hash'):raise ValueError('registered input changed: '+k)
    rows=[json.loads(l) for l in args.inputs.open()]
    hs=[json.loads(l) for l in args.headers.open()];headers={h['path']:h for h in hs}
    if len(rows)!=30733 or len({r['id'] for r in rows})!=30733 or len(headers)!=len(hs):
        raise ValueError('full unique release roster and headers required')
    registry=args.registry.read_bytes()
    processor=Qwen3OmniMoeProcessor.from_pretrained(args.model,local_files_only=True)
    if processor.feature_extractor.hop_length!=160:raise ValueError('registered audio hop required')
    tokenizer=processor.tokenizer;aid=tokenizer.convert_tokens_to_ids('<|audio_pad|>')
    cache={};maxima={};records=[]
    for row in rows:
        inp=row['input'];validate(inp)
        layout=(inp['mode'],tuple((a.get('turn'),a.get('role')) for a in inp['audios']))
        frames=[]
        for a in inp['audios']:
            h=headers[a['path']]
            if h['manifest_sha256']!=a['sha256']:raise ValueError('header audio identity differs')
            frames.append(samples_and_frames(h)[1])
        lengths=_get_feat_extract_output_lengths(torch.tensor(frames)).tolist()
        if layout not in cache:
            prompt=processor.apply_chat_template(messages(inp,['']*len(frames),registry),
                tokenize=False,add_generation_prompt=True)
            ids=tokenizer(prompt,add_special_tokens=False)['input_ids']
            if ids.count(aid)!=len(frames):raise ValueError('audio placeholder count differs')
            cache[layout]=(prompt,len(ids)-len(frames))
        total=cache[layout][1]+sum(lengths)
        record=dict(id=row['id'],tier=row['tier'],mode=inp['mode'],input_tokens=total,
                    frames=frames,audio_tokens=lengths)
        records.append(record)
        if inp['mode'] not in maxima or total>maxima[inp['mode']][0]['input_tokens']:
            maxima[inp['mode']]=(record,row)
    with (args.output/'budgets.jsonl').open('x') as f:
        for record in records:f.write(json.dumps(record)+'\n')
    real=[]
    for mode,(budget,row) in sorted(maxima.items()):
        atomic_json(args.output/'progress.json',dict(phase='real_processor',mode=mode,heartbeat_unix=time.time()))
        waves=[];conversions=[]
        for a in row['input']['audios']:
            path=args.root/a['path']
            if not path.resolve().is_relative_to(args.root.resolve()) or file_hash(path)!=a['sha256']:
                raise ValueError('source waveform path or hash differs')
            wave,conversion=load_sfc_audio(path)
            if len(wave)!=samples_and_frames(headers[a['path']])[0]:raise ValueError('resampling samples differ')
            waves.append(wave);conversions.append(conversion)
        waves=align_to_hop(waves,160)
        prompt=processor.apply_chat_template(messages(row['input'],waves,registry),tokenize=False,add_generation_prompt=True)
        batch=processor(text=[prompt],audio=waves,sampling_rate=16000,return_tensors='pt',padding=True,truncation=False)
        observed=batch['feature_attention_mask'].sum(-1).tolist()
        parts=prompt.split('<|audio_pad|>')
        expanded=parts[0]+''.join('<|audio_pad|>'*n+tail for n,tail in zip(budget['audio_tokens'],parts[1:],strict=True))
        predicted=tokenizer(expanded,add_special_tokens=False)['input_ids']
        if (observed!=budget['frames'] or batch['input_ids'][0].tolist()!=predicted
                or len(predicted)!=budget['input_tokens'] or not torch.isfinite(batch['input_features']).all()):
            raise ValueError('real processor differs from complete budget arithmetic')
        real.append(dict(**budget,conversions=conversions,feature_shape=list(batch['input_features'].shape),
                         token_ids_exact=True,full_frames=True,all_finite=True))
        del batch,waves
    # Token expansion is additive for every distinct actual message layout.
    max_audio=max(max(r['audio_tokens']) for r in records)
    for prompt,text_count in cache.values():
        expanded=prompt.replace('<|audio_pad|>','<|audio_pad|>'*max_audio)
        ids=tokenizer(expanded,add_special_tokens=False)['input_ids']
        if len(ids)!=text_count+prompt.count('<|audio_pad|>')*max_audio:
            raise ValueError('nonadditive audio token expansion')
    atomic_json(args.output/'result.json',dict(status='passed',requests=len(records),layouts=len(cache),
        max_input_tokens=max(r['input_tokens'] for r in records),real_processor_maxima=real,
        exceeded_with_1024_output={str(n):dict(Counter(str(r['tier']) for r in records if r['input_tokens']+1024>n))
                                  for n in (4096,8192,16384,32768)},
        input_hashes={k:file_hash(getattr(args,k)) for k in ('inputs','headers','registry')},
        budgets_sha256=file_hash(args.output/'budgets.jsonl'),
        renderer_sha256=file_hash(inspect.getfile(messages)),processor_sha256=file_hash(inspect.getfile(type(processor))),
        runner_sha256=file_hash(__file__),elapsed_seconds=time.monotonic()-begin,model_loaded=False,annotations_read=False,
        scope='All 30733 fixed native records budgeted with real tokenizer; three mode maxima processed with complete verified waveforms and explicit hop alignment. No full-population waveform validation, vLLM engine/NPU memory, scoring or model inference.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','model','inputs','headers','registry','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('inputs','headers','registry'):p.add_argument('--'+k+'-hash',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:run(a)
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0))

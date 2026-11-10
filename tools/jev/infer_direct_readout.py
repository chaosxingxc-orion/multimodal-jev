#!/usr/bin/env python3
"""Frozen direct audio finite-label readout; one model, no transcription."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.consumption import canonical,validate_request
from multimodal_jev.direct_readout import prompt,label_tokens,decision
from multimodal_jev.end_to_end import require_audio_task_route
from multimodal_jev.policy_identity import checkpoint_identity


def run(cfg,output):
    require_audio_task_route(cfg)
    if cfg.get('parameters_frozen') is not True or cfg.get('adapter') is not None:
        raise ValueError('frozen official checkpoint required')
    if cfg['split']!='train' or cfg['audio_condition'] not in ('actual','zeros'):
        raise ValueError('registered TRAIN actual/zero-waveform probe required')
    import torch
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.audio_policy import load_audio
    torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
    checkpoint=Path(cfg['audio_model'])
    if (checkpoint/'EXPORT.json').exists() or (checkpoint/'adapter_config.json').exists():
        raise ValueError('task-trained checkpoint excluded')
    policy,files=checkpoint_identity(checkpoint)
    with Path(cfg['requests']).open() as handle:rows=[validate_request(json.loads(line)) for line in handle]
    groups=set(cfg['source_groups'])
    if len(groups)!=len(cfg['source_groups']) or not groups<={r['source_group'] for r in rows}:
        raise ValueError('invalid source groups')
    rows=[r for r in rows if r['source_group'] in groups]
    if not rows or len({r['id'] for r in rows})!=len(rows) or any(r['split']!=cfg['split'] for r in rows):
        raise ValueError('unique TRAIN population required')
    identity={'config':cfg,'requests_sha256':file_hash(cfg['requests']),
              'instruction_sha256':file_hash(cfg['instruction']),'models':{'audio_model':{'policy_id':policy,'files':files}},
              'runner_sha256':file_hash(__file__),'source_sha256':{name:file_hash(Path(__file__).parents[2]/'src/multimodal_jev'/name)
                for name in ('direct_readout.py','end_to_end.py','audio_policy.py','asr.py','consumption.py','task_schema.py')}}
    if (output/'identity.json').exists() and json.loads((output/'identity.json').read_text())!=identity:
        raise ValueError('resume identity changed')
    atomic_json(output/'identity.json',identity);atomic_json(output/'selection.json',{'ids':[r['id'] for r in rows]})
    cache=output/'records';cache.mkdir(exist_ok=True)
    start=time.monotonic()
    atomic_json(output/'progress.json',{'phase':'audio_model_load','heartbeat_unix':time.time()})
    processor,model=load_runtime(str(checkpoint),cfg['device'],cfg['dtype']);model.requires_grad_(False)
    instruction=Path(cfg['instruction']).read_text();records=[]
    for index,row in enumerate(rows):
        if time.monotonic()-start>cfg['wall_seconds']:raise TimeoutError('probe time limit')
        if file_hash(row['input']['audio'])!=row['audio_sha256']:raise ValueError('audio changed')
        wave=load_audio(row['input']['audio'])
        if cfg['audio_condition']=='zeros':
            import numpy as np
            wave=np.zeros_like(wave)
        system,options=prompt(row,instruction);ids=label_tokens(processor.tokenizer,options)
        messages=[{'role':'system','content':[{'type':'text','text':'English\n'+system}]},
                  {'role':'user','content':[{'type':'audio','audio':wave}]}]
        rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        digest=hashlib.sha256(rendered.encode()).hexdigest()
        path=cache/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json')
        if path.exists():
            record=json.loads(path.read_text())
            if record['example_id']!=row['id'] or record['prompt_sha256']!=digest:
                raise ValueError('cached readout identity changed')
        else:
            batch=dict(processor(text=[rendered],audio=[wave],sampling_rate=16000,return_tensors='pt'))
            if batch['input_ids'].shape[-1]>cfg['max_context_tokens']:raise ValueError('context overflow; no truncation')
            batch={k:v.to(device=model.device,dtype=model.dtype if v.is_floating_point() else v.dtype) for k,v in batch.items()}
            t=time.monotonic()
            with torch.inference_mode():
                result=model(**batch,use_cache=False,logits_to_keep=1)
                scores=result.logits[0,-1,ids].float().cpu().tolist()
            selected=decision(options,scores)
            record={'example_id':row['id'],'replicate':0,'source_group':row['source_group'],'split':row['split'],
                    'request_sha256':hashlib.sha256(canonical(row)).hexdigest(),'prompt_sha256':digest,
                    'raw_text':json.dumps({'value':selected['value']},separators=(',',':')),
                    'value':selected['value'],'format_valid':True,'schema_valid':True,
                    'alternatives':options,'label_token_ids':ids,**selected,
                    'input_tokens':int(batch['input_ids'].shape[-1]),'seconds':time.monotonic()-t,
                    'format_enforced_by_readout':True,'audio_condition':cfg['audio_condition']}
            atomic_json(path,record)
            del result,batch
        records.append(record)
        atomic_json(output/'progress.json',{'phase':'audio_label_forward','done':index+1,'total':len(rows),
                    'elapsed_seconds':time.monotonic()-start,'heartbeat_unix':time.time()})
    if any(p.requires_grad for p in model.parameters()) or checkpoint_identity(checkpoint)[0]!=policy:
        raise ValueError('model frozen identity changed')
    pred=output/'predictions.jsonl';pred.write_bytes(b''.join(map(canonical,records)))
    atomic_json(output/'result.json',{'status':'passed','requests':len(rows),'predictions':len(records),
                 'predictions_sha256':file_hash(pred),'weight_updates':0,'model_files_unchanged':True,
                 'elapsed_seconds':time.monotonic()-start,'audio_condition':cfg['audio_condition'],
                 'scope':'direct audio candidate-normalized next-token readout; output format forced; no JSON generation, transcript cascade, calibration or efficacy claim'})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.output/'result.json').exists():raise FileExistsError('completed run retained')
        try:run(json.loads(args.config.read_text()),args.output)
        except BaseException as exc:
            atomic_json(args.output/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()});raise
        atomic_json(args.output/'terminal.json',{'exit_code':0,'finished_unix':time.time()})

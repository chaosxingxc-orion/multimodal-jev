#!/usr/bin/env python3
"""Frozen HF reference inference; consumes label-free requests, never gold."""
import argparse
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.frozen_evidence import load_evidence
from multimodal_jev.frozen_contract import FrozenContract


def run(config, output):
    import torch
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.audio_policy import load_audio
    route=config['route']
    if route not in {'audio-direct','asr-text'}: raise ValueError('unsupported frozen route')
    if config.get('adapter') is not None or config.get('parameters_frozen') is not True:
        raise ValueError('frozen official checkpoints only')
    sampling=config['sampling']
    if sampling['temperature']<0 or sampling['n']<1: raise ValueError('invalid sampling')
    if sampling['temperature']>0 and (config['split']!='train' or config['purpose']!='tfrl-learning'):
        raise ValueError('stochastic sampling only for TFRL learning')
    contract=FrozenContract(config)
    with Path(config['requests']).open() as handle:
        rows=[contract.validate(json.loads(line)) for line in handle]
    if not rows or len({r['id'] for r in rows})!=len(rows): raise ValueError('unique nonempty requests required')
    if any(r['split']!=config['split'] for r in rows): raise ValueError('split mismatch')
    if config.get('source_groups') is not None:
        selected=set(config['source_groups'])
        if len(selected)!=len(config['source_groups']) or not selected <= {r['source_group'] for r in rows}:
            raise ValueError('invalid source group selection')
        rows=[r for r in rows if r['source_group'] in selected]
    instruction=Path(config['instruction']).read_text()
    if not instruction.strip() or len(instruction)>12000: raise ValueError('instruction budget exceeded')
    torch.set_num_threads(4);torch.manual_seed(config['seed'])
    identities={}
    for name in ('audio_model','text_model'):
        if name in config:
            root=Path(config[name])
            if (root/'EXPORT.json').exists() or (root/'adapter_config.json').exists():
                raise ValueError('task-trained checkpoint excluded')
            policy,files=checkpoint_identity(root)
            identities[name]={'policy_id':policy,'files':files}
    identity={'config':config,'requests_sha256':file_hash(config['requests']),
        'instruction_sha256':file_hash(config['instruction']),'models':identities,
        'runner_sha256':file_hash(__file__),
        'source_sha256':{name:file_hash(Path(__file__).parents[2]/'src/multimodal_jev'/name)
            for name in ('consumption.py','asr.py','audio_policy.py','policy_evaluation.py','answers.py','task_schema.py','frozen_evidence.py','frozen_contract.py','slurp_native.py','teacher_candidates.py')}}
    if contract.taxonomy_sha256 is not None:identity['taxonomy_sha256']=contract.taxonomy_sha256
    if (output/'identity.json').exists() and json.loads((output/'identity.json').read_text())!=identity:
        raise ValueError('resume identity changed')
    atomic_json(output/'identity.json',identity)
    atomic_json(output/'selection.json',{'ids':[r['id'] for r in rows]})
    cache=output/'records';cache.mkdir(exist_ok=True)
    asr_cache=output/'transcriptions';asr_cache.mkdir(exist_ok=True)
    start=time.monotonic();texts={};records=[]
    def progress(phase,done,total):
        atomic_json(output/'progress.json',{'phase':phase,'done':done,'total':total,
            'elapsed_seconds':time.monotonic()-start,'heartbeat_unix':time.time()})
    def move(batch):
        return {k:v.to(device=model.device,dtype=model.dtype if v.is_floating_point() else v.dtype) for k,v in batch.items()}
    def check_length(batch,limit):
        if batch['input_ids'].shape[-1]+limit>config['max_context_tokens']:
            raise ValueError('context overflow; truncation prohibited')
    def check_time():
        if time.monotonic()-start>config['wall_seconds']:raise TimeoutError('inference walltime exceeded')
    reuse=config.get('asr_evidence')
    if reuse is not None:
        pinned=load_evidence(reuse,rows,config,identities)
        for audio_hash,record in pinned.items():
            path=asr_cache/(audio_hash+'.json')
            if path.exists() and json.loads(path.read_text())!=record:
                raise ValueError('local transcript differs from pinned evidence')
            atomic_json(path,record)
        progress('reused_pinned_transcriptions',len(pinned),len(pinned))
    else:
        progress('audio_model_load',0,len(rows))
        processor,model=load_runtime(config['audio_model'],config['device'],config['dtype'])
        model.requires_grad_(False)
    if route=='asr-text':
        unique={r['audio_sha256']:r for r in rows}
        for i,(audio_hash,row) in enumerate(unique.items()):
            check_time()
            if file_hash(row['input']['audio'])!=audio_hash:raise ValueError('audio changed')
            path=asr_cache/(audio_hash+'.json')
            if path.exists():
                record=json.loads(path.read_text())
                if record['audio_sha256']!=audio_hash:raise ValueError('cached audio changed')
            else:
                wave=load_audio(row['input']['audio'])
                batch=move(processor.apply_transcription_request(wave,language='English',return_tensors='pt',sampling_rate=16000))
                check_length(batch,config['asr_max_new_tokens']);t=time.monotonic()
                with torch.inference_mode():tokens=model.generate(**batch,do_sample=False,max_new_tokens=config['asr_max_new_tokens'])
                ids=tokens[0,batch['input_ids'].shape[-1]:].tolist()
                raw=processor.tokenizer.decode(ids,skip_special_tokens=True)
                record={'audio_sha256':audio_hash,'transcription':processor.extract_transcription(raw),
                    'raw_text':raw,'token_ids':ids,'seconds':time.monotonic()-t,
                    'length_limited':len(ids)>=config['asr_max_new_tokens']}
                atomic_json(path,record)
            progress('transcription',i+1,len(unique))
        texts={r['id']:json.loads((asr_cache/(r['audio_sha256']+'.json')).read_text())['transcription'] for r in rows}
        if reuse is None:del model,processor
        gc.collect()
        if config['device'].startswith('npu'):torch.npu.empty_cache()
        progress('text_model_load',0,len(rows))
        tokenizer=AutoTokenizer.from_pretrained(config['text_model'],local_files_only=True)
        model,loading=Qwen3ForCausalLM.from_pretrained(config['text_model'],local_files_only=True,
            torch_dtype=getattr(torch,config['dtype']),attn_implementation='eager',output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('non-exact text checkpoint load')
        model=model.to(config['device']).eval().requires_grad_(False)
    for index,row in enumerate(rows):
        if file_hash(row['input']['audio'])!=row['audio_sha256']:raise ValueError('audio changed')
        if route=='audio-direct':
            wave=load_audio(row['input']['audio'])
            system=instruction+'\nDecision definition:\n'+json.dumps(row['input']['schema'],sort_keys=True)
            messages=[{'role':'system','content':[{'type':'text','text':'English\n'+system}]},
                {'role':'user','content':[{'type':'audio','audio':wave}]}]
            prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            batch=move(processor(text=[prompt],audio=[wave],sampling_rate=16000,return_tensors='pt'))
            decoder=processor.tokenizer
        else:
            prompt=tokenizer.apply_chat_template(contract.messages(row,texts[row['id']],instruction),
                tokenize=False,add_generation_prompt=True,enable_thinking=False)
            batch=move(tokenizer(prompt,return_tensors='pt',add_special_tokens=False));decoder=tokenizer
        check_length(batch,config['max_new_tokens'])
        prompt_sha=hashlib.sha256(prompt.encode()).hexdigest()
        for replicate in range(sampling['n']):
            check_time()
            key=row['id']+':'+str(replicate)
            path=cache/(hashlib.sha256(key.encode()).hexdigest()+'.json')
            if path.exists():
                record=json.loads(path.read_text())
                if record['example_id']!=row['id'] or record['prompt_sha256']!=prompt_sha or record['replicate']!=replicate:
                    raise ValueError('cached prediction changed')
            else:
                seed=config['seed']+replicate+int(hashlib.sha256(row['id'].encode()).hexdigest()[:7],16)
                torch.manual_seed(seed);kwargs={}
                if sampling['temperature']>0:kwargs=dict(temperature=sampling['temperature'],top_p=1.,top_k=0)
                t=time.monotonic()
                with torch.inference_mode():tokens=model.generate(**batch,do_sample=sampling['temperature']>0,
                    max_new_tokens=config['max_new_tokens'],**kwargs)
                ids=tokens[0,batch['input_ids'].shape[-1]:].tolist();raw=decoder.decode(ids,skip_special_tokens=True)
                record=contract.prediction(row,raw)
                record.update(replicate=replicate,source_group=row['source_group'],split=row['split'],
                    prompt_sha256=prompt_sha,request_sha256=hashlib.sha256(canonical(row)).hexdigest(),
                    input_tokens=int(batch['input_ids'].shape[-1]),token_ids=ids,seconds=time.monotonic()-t,
                    length_limited=len(ids)>=config['max_new_tokens'])
                atomic_json(path,record)
            records.append(record)
        progress('decision',index+1,len(rows))
    if any(p.requires_grad for p in model.parameters()):raise ValueError('model not frozen')
    for key,stored in identities.items():
        if checkpoint_identity(config[key])[0]!=stored['policy_id']:raise ValueError('base model changed')
    path=output/'predictions.jsonl';tmp=path.with_suffix('.tmp')
    tmp.write_bytes(b''.join(canonical(r) for r in records));tmp.replace(path)
    atomic_json(output/'result.json',{'status':'passed','requests':len(rows),'predictions':len(records),
        'format_valid':sum(r['format_valid'] for r in records),'schema_valid':sum(r['schema_valid'] for r in records),
        'length_limited':sum(r['length_limited'] for r in records),'route':route,
        'predictions_sha256':file_hash(path),'weight_updates':0,'model_files_unchanged':True,
        'asr_evidence':reuse,
        'elapsed_seconds':time.monotonic()-start,
        'scope':'frozen HF reference inference; separate scoring required; no vLLM parity or training claim'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.output/'result.json').exists():raise FileExistsError('completed run retained')
        try:run(json.loads(args.config.read_text()),args.output)
        except BaseException as exc:
            atomic_json(args.output/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()});raise
        atomic_json(args.output/'terminal.json',{'exit_code':0,'finished_unix':time.time()})

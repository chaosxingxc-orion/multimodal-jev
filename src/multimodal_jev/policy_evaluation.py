"""Resumable, label-blind generation on a frozen development manifest."""
from __future__ import annotations
import json
from pathlib import Path
import time
from .baseline import atomic_json,file_hash
from .task_schema import JevExample
from .answers import parse_answer,AnswerError


def prediction_record(example_id,text):
    record={'example_id':example_id,'raw_text':text}
    try:
        record.update(parse_answer(text));record['format_valid']=True
    except AnswerError as exc:
        record.update(format_valid=False,format_error=str(exc))
    return record


def run(config,output):
    import torch
    from .asr import load_runtime
    from .audio_policy import hf_prompt,load_audio
    from .evaluate import evaluate
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(config['cpu_threads']);torch.manual_seed(config['seed'])
    manifest=Path(config['manifest'])
    adapter=Path(config['adapter']) if config.get('adapter') else None
    if adapter is not None and not (adapter/'adapter_model.safetensors').is_file():
        raise FileNotFoundError('completed adapter required for evaluation')
    identity={'config':config,'manifest_sha256':file_hash(manifest),
              'weights_sha256':file_hash(Path(config['model_path'])/'model.safetensors'),
              'adapter_sha256':({p.name:file_hash(p) for p in adapter.iterdir() if p.is_file()} if adapter else None),
              'source_sha256':{name:file_hash(Path(__file__).with_name(name)) for name in
                               ('policy_evaluation.py','audio_policy.py','asr.py','answers.py','evaluate.py','reward.py')}}
    if (output/'identity.json').exists() and json.loads((output/'identity.json').read_text())!=identity:
        raise ValueError('evaluation resume identity changed')
    if (output/'metrics.json').exists():raise FileExistsError('completed evaluation')
    atomic_json(output/'identity.json',identity)
    examples=[JevExample.from_record(json.loads(line)) for line in manifest.open()]
    if any(ex.split!='valid' for ex in examples):raise ValueError('development runner admits valid only')
    if len({ex.example_id for ex in examples})!=len(examples):raise ValueError('duplicate evaluation IDs')
    cache=output/'records';cache.mkdir(exist_ok=True)
    processor,model=load_runtime(config['model_path'],config['device'],config['dtype'])
    if adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,adapter,is_trainable=False)
    model.eval();start=time.monotonic()
    records=[]
    for index,ex in enumerate(examples):
        import hashlib
        path=cache/(hashlib.sha256(ex.example_id.encode()).hexdigest()+'.json')
        if path.exists():
            record=json.loads(path.read_text())
            if record['example_id']!=ex.example_id:raise ValueError('cache ID mismatch')
        else:
            if file_hash(ex.audio)!=ex.metadata['audio_sha256']:raise ValueError('audio changed')
            batch=hf_prompt(processor,ex,load_audio(ex.audio))
            batch={k:v.to(device=model.device,dtype=model.dtype if v.is_floating_point() else v.dtype) for k,v in batch.items()}
            t=time.monotonic()
            with torch.inference_mode():
                tokens=model.generate(**batch,max_new_tokens=config['max_new_tokens'],do_sample=False)
            response=tokens[0,batch['input_ids'].shape[-1]:].tolist()
            text=processor.tokenizer.decode(response,skip_special_tokens=True)
            record=prediction_record(ex.example_id,text)
            record.update(token_ids=response,generation_seconds=time.monotonic()-t,
                reached_length_limit=len(response)>=config['max_new_tokens'],input_tokens=batch['input_ids'].shape[-1])
            atomic_json(path,record)
        records.append(record)
        atomic_json(output/'progress.json',{'done':index+1,'total':len(examples),'heartbeat_unix':time.time(),
                    'elapsed_this_session_seconds':time.monotonic()-start})
    predictions=output/'predictions.jsonl'
    predictions.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    metrics=evaluate(manifest,predictions)
    metrics.update(valid_json_objects=sum(r['format_valid'] for r in records),config=config,
                   scope='fixed development subset; no test evaluation or final generalization claim')
    atomic_json(output/'metrics.json',metrics)
    return metrics

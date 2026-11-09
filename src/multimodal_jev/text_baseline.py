"""Matched ASR-transcript/reference-transcript decision controls.

References are used only by the explicitly labeled reference-text condition.
The ASR condition admits only saved model transcriptions, never its references.
"""
import hashlib
import json
from pathlib import Path
import time
from .audio_policy import INSTRUCTION
from .baseline import atomic_json,file_hash
from .policy_evaluation import prediction_record
from .policy_identity import checkpoint_identity
from .task_schema import JevExample,render_schema


def messages(example,transcription):
    if not isinstance(transcription,str):raise ValueError('transcription must be text')
    return [
        {'role':'system','content':'English\n'+INSTRUCTION.replace('audio evidence','transcript evidence')+render_schema(example.schema)},
        {'role':'user','content':json.dumps({'transcription':transcription},ensure_ascii=False)}]


def load_evidence(config):
    manifest=Path(config['manifest']);manifest_sha=file_hash(manifest)
    examples=[JevExample.from_record(json.loads(line)) for line in manifest.open()]
    if not examples or any(ex.split!='valid' for ex in examples):
        raise ValueError('nonempty valid-only development manifest required')
    by_id={ex.example_id:ex for ex in examples}
    if len(by_id)!=len(examples):raise ValueError('duplicate manifest example')
    if config['input_mode']=='reference-text':
        if config.get('asr_run') is not None:raise ValueError('reference condition must not use ASR run')
        texts={ex.example_id:ex.metadata.get('transcript_reference') for ex in examples}
        if any(not isinstance(t,str) or not t.strip() for t in texts.values()):
            raise ValueError('reference transcript missing')
        return examples,texts,{'mode':'reference-text','manifest_sha256':manifest_sha}
    if config['input_mode']!='asr-text':raise ValueError('unknown input mode')
    run=Path(config['asr_run'])
    terminal=json.loads((run/'terminal.json').read_text());result=json.loads((run/'result.json').read_text())
    identity=json.loads((run/'identity.json').read_text());predictions=run/'predictions.jsonl'
    if terminal['exit_code']!=0 or result['status']!='passed':raise ValueError('completed ASR required')
    if identity['manifest_sha256']!=manifest_sha:raise ValueError('ASR manifest mismatch')
    if result['predictions_sha256']!=file_hash(predictions):raise ValueError('ASR predictions changed')
    texts={};audios={}
    with predictions.open() as handle:
        for line in handle:
            row=json.loads(line);uid=row['example_id']
            if uid not in by_id or uid in texts:raise ValueError('ASR IDs missing, duplicated or unexpected')
            audio_hash=by_id[uid].metadata['audio_sha256']
            if row['audio_sha256']!=audio_hash:raise ValueError('ASR audio identity mismatch')
            text=row['transcription']
            if not isinstance(text,str):raise ValueError('ASR transcription must be text')
            if audio_hash in audios and audios[audio_hash]!=text:
                raise ValueError('same audio has inconsistent ASR evidence')
            texts[uid]=text;audios[audio_hash]=text
    if set(texts)!=set(by_id) or result['records']!=len(examples):
        raise ValueError('complete aligned ASR predictions required')
    return examples,texts,{'mode':'asr-text','manifest_sha256':manifest_sha,
        'asr_identity_sha256':file_hash(run/'identity.json'),
        'asr_terminal_sha256':file_hash(run/'terminal.json'),
        'asr_result_sha256':file_hash(run/'result.json'),
        'asr_predictions_sha256':result['predictions_sha256'],'asr_run':str(run)}


def run(config,output):
    import torch
    from transformers import AutoTokenizer,Qwen3ForCausalLM
    from .evaluate import evaluate
    if config['device']!='cpu' or config['dtype']!='float32':
        raise ValueError('registered text baseline requires CPU float32')
    if config.get('enable_thinking') is not False:raise ValueError('non-thinking protocol required')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'metrics.json').exists():raise FileExistsError('completed text evaluation')
    examples,evidence,evidence_identity=load_evidence(config)
    policy,files=checkpoint_identity(config['model_path'])
    identity={'config':config,'manifest_sha256':file_hash(config['manifest']),
        'policy_id':policy,'checkpoint_files':files,'evidence':evidence_identity,
        'source_sha256':{name:file_hash(Path(__file__).with_name(name)) for name in
            ('text_baseline.py','policy_evaluation.py','audio_policy.py','evaluate.py','reward.py','answers.py','task_schema.py')}}
    path=output/'identity.json'
    if path.exists() and json.loads(path.read_text())!=identity:raise ValueError('text evaluation identity changed')
    atomic_json(path,identity)
    torch.set_num_threads(config['cpu_threads']);torch.manual_seed(config['seed'])
    atomic_json(output/'progress.json',{'phase':'model_load','done':0,'total':len(examples),'heartbeat_unix':time.time()})
    tokenizer=AutoTokenizer.from_pretrained(config['model_path'],local_files_only=True,trust_remote_code=False)
    model,loading=Qwen3ForCausalLM.from_pretrained(config['model_path'],local_files_only=True,
        torch_dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
    if any(loading.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError(f'non-exact text model load: {loading}')
    model=model.to('cpu').eval();cache=output/'records';cache.mkdir(exist_ok=True)
    start=time.monotonic();records=[]
    for index,example in enumerate(examples):
        prompt=tokenizer.apply_chat_template(messages(example,evidence[example.example_id]),
                    tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prompt_sha=hashlib.sha256(prompt.encode()).hexdigest()
        path=cache/(hashlib.sha256(example.example_id.encode()).hexdigest()+'.json')
        if path.exists():
            record=json.loads(path.read_text())
            if record['example_id']!=example.example_id or record['prompt_sha256']!=prompt_sha:
                raise ValueError('text cache identity changed')
        else:
            inputs=tokenizer(prompt,return_tensors='pt',add_special_tokens=False)
            if inputs['input_ids'].shape[-1]+config['max_new_tokens']>model.config.max_position_embeddings:
                raise ValueError('context length exceeded; truncation prohibited')
            started=time.monotonic()
            with torch.inference_mode():
                generated=model.generate(**inputs,do_sample=False,max_new_tokens=config['max_new_tokens'],
                                         pad_token_id=tokenizer.pad_token_id)
            ids=generated[0,inputs['input_ids'].shape[-1]:].tolist()
            raw=tokenizer.decode(ids,skip_special_tokens=True)
            record=prediction_record(example.example_id,raw)
            record.update(prompt_sha256=prompt_sha,input_mode=config['input_mode'],
                transcript_sha256=hashlib.sha256(evidence[example.example_id].encode()).hexdigest(),
                token_ids=ids,input_tokens=int(inputs['input_ids'].shape[-1]),
                generation_seconds=time.monotonic()-started,reached_length_limit=len(ids)>=config['max_new_tokens'])
            atomic_json(path,record)
        records.append(record)
        atomic_json(output/'progress.json',{'phase':'generation','done':index+1,'total':len(examples),
            'heartbeat_unix':time.time(),'elapsed_this_session_seconds':time.monotonic()-start})
    predictions=output/'predictions.jsonl';temp=predictions.with_suffix('.tmp')
    temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records));temp.replace(predictions)
    metrics=evaluate(Path(config['manifest']),predictions)
    metrics.update(config=config,valid_json_objects=sum(r['format_valid'] for r in records),
                   scope='fixed development text-evidence control; no acoustic reasoning, training or RL efficacy claim')
    atomic_json(output/'metrics.json',metrics)
    atomic_json(output/'result.json',{'status':'passed','records':len(records),
        'predictions_sha256':file_hash(predictions),'metrics_sha256':file_hash(output/'metrics.json'),
        'input_mode':config['input_mode'],'policy_id':policy})
    return metrics

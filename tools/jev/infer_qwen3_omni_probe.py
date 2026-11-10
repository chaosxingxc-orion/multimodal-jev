#!/usr/bin/env python3
"""Frozen official local Qwen3-Omni, audio waveform to task JSON via vLLM."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.end_to_end import require_audio_task_route
from multimodal_jev.frozen_contract import FrozenContract
from multimodal_jev.omni_probe_contract import select_learning
from multimodal_jev.omni_test_contract import select_test
from multimodal_jev.omni_slurp_contract import validate_runtime
from multimodal_jev.waveform_control import condition_audio


def verify_assets(cfg):
    audit=json.loads(Path(cfg['asset_audit']).read_text());root=Path(cfg['audio_model'])
    if (audit['status']!='verified_inference_assets' or audit['model']!=str(root)
            or audit['repository']!='Qwen/Qwen3-Omni-30B-A3B-Instruct'
            or audit['revision']!='26291f793822fb6be9555850f06dfe95f2d7e695'
            or not audit['all_publisher_files_present'] or not audit['files']):
        raise ValueError('complete pinned original model required')
    for row in audit['files']:
        if file_hash(root/row['file'])!=row['sha256']:raise ValueError('model file changed: '+row['file'])
    return audit


def run(cfg,out,*,population_selector=select_learning):
    require_audio_task_route(cfg)
    validate_runtime(cfg)
    allowed_split = 'test' if population_selector is select_test else 'train'
    if (cfg['parameters_frozen'] is not True or cfg['split']!=allowed_split
            or cfg.get('adapter') is not None):
        raise ValueError('registered population and frozen official base only')
    start=time.monotonic();atomic_json(out/'progress.json',dict(phase='verifying_model',heartbeat_unix=time.time()))
    audit=verify_assets(cfg)
    contract=FrozenContract(cfg)
    rows=[contract.validate(json.loads(line)) for line in Path(cfg['requests']).open()]
    rows=population_selector(cfg,rows)
    condition=cfg.get('audio_condition','actual')
    identity=dict(config=cfg,requests_sha256=file_hash(cfg['requests']),instruction_sha256=file_hash(cfg['instruction']),
        asset_audit_sha256=file_hash(cfg['asset_audit']),model_files=audit['files'],runner_sha256=file_hash(__file__),
        source_sha256={n:file_hash(Path(__file__).parents[2]/'src/multimodal_jev'/n) for n in
                      ['frozen_contract.py','consumption.py','task_schema.py','answers.py','audio_policy.py','end_to_end.py','omni_probe_contract.py','omni_test_contract.py','waveform_control.py','omni_slurp_contract.py','slurp_native.py']},
        versions={p:importlib.metadata.version(p) for p in ['torch','torch-npu','transformers','vllm','vllm-ascend']})
    if population_selector is not select_learning:
        import inspect
        identity['population_selector_sha256']=file_hash(inspect.getfile(population_selector))
        if 'selection_protocol_sha256' in cfg:
            identity['selection_protocol_sha256']=cfg['selection_protocol_sha256']
    if contract.taxonomy_sha256 is not None:
        identity['taxonomy_sha256']=contract.taxonomy_sha256
    if population_selector is select_test:
        protocol = json.loads(Path(cfg['test_protocol']).read_text())
        if any(identity[k] != v for k,v in protocol['actor_identity'].items()):
            raise ValueError('TEST actor differs from completed selection')
        identity['test_protocol_sha256'] = cfg['test_protocol_sha256']
    atomic_json(out/'identity.json',identity)
    atomic_json(out/'selection.json',{'ids':[r['id'] for r in rows]})
    from transformers import Qwen3OmniMoeProcessor
    from multimodal_jev.audio_policy import load_audio
    from vllm import LLM,SamplingParams
    processor=Qwen3OmniMoeProcessor.from_pretrained(cfg['audio_model'],local_files_only=True)
    atomic_json(out/'progress.json',dict(phase='engine_load',heartbeat_unix=time.time()))
    llm=LLM(model=cfg['audio_model'],dtype='bfloat16',tensor_parallel_size=2,seed=17,
        max_model_len=4096,max_num_seqs=1,max_num_batched_tokens=4096,
        gpu_memory_utilization=.85,enforce_eager=True,trust_remote_code=False,
        limit_mm_per_prompt={'audio':1,'image':0,'video':0},enable_prefix_caching=False)
    params=SamplingParams(temperature=0.,max_tokens=cfg['max_new_tokens'],seed=17)
    instruction=Path(cfg['instruction']).read_text();records=[]
    cache=out/'records';cache.mkdir()
    for i,row in enumerate(rows):
        if file_hash(row['input']['audio'])!=row['audio_sha256']:raise ValueError('audio changed')
        wave=condition_audio(load_audio(row['input']['audio']),condition,cfg['split'])
        if len(wave)>processor.feature_extractor.chunk_length*16000:raise ValueError('audio truncation prohibited')
        definition=contract.audio_instruction(row,instruction)
        messages=[{'role':'system','content':'You are a helpful assistant.'},
                  {'role':'user','content':[{'type':'audio','audio':wave},{'type':'text','text':definition}]}]
        prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        t=time.monotonic()
        response=llm.generate([{'prompt':prompt,'multi_modal_data':{'audio':(wave,16000)}}],params,use_tqdm=False)
        if len(response)!=1 or len(response[0].outputs)!=1:raise ValueError('single output required')
        answer=response[0].outputs[0];record=contract.prediction(row,answer.text)
        record.update(replicate=0,source_group=row['source_group'],split=cfg['split'],audio_condition=condition,
            request_sha256=hashlib.sha256(canonical(row)).hexdigest(),prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            input_tokens=len(response[0].prompt_token_ids),token_ids=list(answer.token_ids),
            finish_reason=answer.finish_reason,length_limited=answer.finish_reason=='length',seconds=time.monotonic()-t)
        atomic_json(cache/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json'),record);records.append(record)
        atomic_json(out/'progress.json',dict(phase='decision',done=i+1,total=len(rows),heartbeat_unix=time.time()))
    atomic_json(out/'progress.json',dict(phase='engine_shutdown',done=len(rows),total=len(rows),heartbeat_unix=time.time()))
    llm.llm_engine.engine_core.shutdown(timeout=30)
    atomic_json(out/'progress.json',dict(phase='final_model_verification',done=len(rows),total=len(rows),heartbeat_unix=time.time()))
    verify_assets(cfg)
    path=out/'predictions.jsonl';path.write_bytes(b''.join(canonical(r) for r in records))
    atomic_json(out/'result.json',dict(status='passed',requests=len(rows),predictions=len(records),audio_condition=condition,
        format_valid=sum(r['format_valid'] for r in records),schema_valid=sum(r['schema_valid'] for r in records),
        length_limited=sum(r['length_limited'] for r in records),predictions_sha256=file_hash(path),weight_updates=0,
        model_files_unchanged=True,elapsed_seconds=time.monotonic()-start,
        scope='direct frozen audio-to-task vLLM NPU registered '+cfg['split']+' population; no HF parity claim'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=True)
    with (a.output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.output/'identity.json').exists():raise FileExistsError('preserve prior attempt; no blind retry')
        try:run(json.loads(a.config.read_text()),a.output)
        except BaseException as exc:
            atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))

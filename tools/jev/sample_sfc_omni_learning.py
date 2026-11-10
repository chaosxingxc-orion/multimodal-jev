#!/usr/bin/env python3
"""Independent frozen K=2 TRAIN sampling; never modifies the greedy baseline.

--preflight-only validates actual vLLM SamplingParams and all input identities
without loading model weights. Runtime always uses a new output directory.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.omni_actor_batch import prepare, evidence
from multimodal_jev.omni_rollout_records import capture, sampling_kwargs, validate_sampling
from multimodal_jev.sfc_audio import load_sfc_audio
from multimodal_jev.sfc_jev_runtime import INSTRUCTION, vllm_request
from multimodal_jev.sfc_reward_batch import digest
from multimodal_jev.sfc_teacher_feedback import validate_partition
from infer_sfc_jev_omni_probe import select_inputs
from infer_qwen3_omni_probe import verify_assets


def load_inputs(config):
    if set(config) != {'baseline_config','baseline_config_sha256','partition','partition_sha256',
                       'group_size','seed','max_tokens'}:
        raise ValueError('exact independent sampling config required')
    if (config['group_size'],config['seed'],config['max_tokens']) != (2,17,1024):
        raise ValueError('registered complete 78-decision K=2 sampling required')
    for key in ('baseline_config','partition'):
        if file_hash(config[key]) != config[key+'_sha256']:
            raise ValueError('registered input changed: '+key)
    base = json.loads(Path(config['baseline_config']).read_text())
    rows, identity, checks = select_inputs(base)
    partition = json.loads(Path(config['partition']).read_text())
    all_rows = [json.loads(line) for line in (Path(identity['release'])/'train_id.requests.jsonl').open()]
    validate_partition(all_rows, partition)
    if not {r['source_group'] for r in rows} <= set(partition['learning']):
        raise ValueError('sampling restricted to complete learning conversations')
    return base, rows, identity, checks


def run(config, out, preflight_only):
    start = time.monotonic()
    base, rows, preflight, checks = load_inputs(config)
    versions={k:importlib.metadata.version(k) for k in base['versions']}
    if versions != base['versions']:
        raise ValueError('pinned engine versions required')
    for name, pin in preflight['processor_assets'].items():
        if file_hash(Path(base['audio_model'])/name) != pin:
            raise ValueError('processor assets changed')
    from transformers import Qwen3OmniMoeProcessor
    from vllm import SamplingParams
    import torch
    torch.set_num_threads(2)
    processor=Qwen3OmniMoeProcessor.from_pretrained(base['audio_model'],local_files_only=True)
    atomic_json(out/'progress.json',dict(phase='processor',heartbeat_unix=time.time()))
    prepared=prepare(processor,rows)
    for key, pin in evidence(prepared).items():
        if any(checks[key][name]!=value for name,value in pin.items()):
            raise ValueError('actor inputs differ from registered processor preflight')
        if len(prepared[key]['input_ids'])+1024>4096:
            raise ValueError('full response budget does not fit')
    params=SamplingParams(**sampling_kwargs(2,17))
    effective=validate_sampling(params,2,17)
    engine_args=dict(model=base['audio_model'],dtype='bfloat16',tensor_parallel_size=2,seed=17,
        max_model_len=4096,max_num_seqs=1,max_num_batched_tokens=4096,
        gpu_memory_utilization=.85,enforce_eager=True,trust_remote_code=False,
        limit_mm_per_prompt={'audio':2,'image':0,'video':0},enable_prefix_caching=False,
        logprobs_mode='raw_logprobs',generation_config='vllm')
    identity=dict(config=config,versions=versions,engine_arguments=engine_args,
        effective_sampling=effective,inputs_sha256=digest(rows),
        instruction_sha256=hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
        prompt_evidence=[dict(example_id=k[0],decision_id=k[1],**v) for k,v in evidence(prepared).items()],
        source_files={str(p.relative_to(Path(__file__).parents[2])):file_hash(p)
            for p in (Path(__file__).parents[2]/'src/multimodal_jev').glob('*.py')},
        runner_sha256=file_hash(__file__),preflight_only=preflight_only)
    atomic_json(out/'identity.json',identity)
    if preflight_only:
        result=dict(status='passed',bundles=len(rows),decisions=len(prepared),sampling=effective,
            model_loaded=False,weight_updates=0,elapsed_seconds=time.monotonic()-start,
            identity_sha256=file_hash(out/'identity.json'),
            scope='Real processor and SamplingParams only; no model/engine generation or probabilities')
        atomic_json(out/'result.json',result)
        return
    atomic_json(out/'progress.json',dict(phase='verify_weights',heartbeat_unix=time.time()))
    assets=verify_assets(base)
    policy=digest(dict(repository=assets['repository'],revision=assets['revision'],files=assets['files']))
    identity.update(policy_sha256=policy,model_files=assets['files'])
    atomic_json(out/'identity.json',identity)
    from vllm import LLM
    atomic_json(out/'progress.json',dict(phase='engine_load',heartbeat_unix=time.time()))
    engine=LLM(**engine_args)
    records=[]
    (out/'requests').mkdir()
    try:
        if (engine.model_config.logprobs_mode!='raw_logprobs'
                or engine.model_config.generation_config!='vllm'):
            raise ValueError('effective engine probability mode differs')
        atomic_json(out/'effective-engine.json',dict(logprobs_mode=engine.model_config.logprobs_mode,
            generation_config=engine.model_config.generation_config))
        for row in rows:
            waves=[]
            for path,pin in zip(row['input']['context']['audios'],row['audio_sha256'],strict=True):
                if file_hash(path)!=pin:raise ValueError('audio changed')
                waves.append(load_sfc_audio(path)[0])
            for d in row['input']['decisions']:
                key=row['id'],d['id'];item=prepared[key]
                request=vllm_request(processor,row,d['id'],waves)
                if hashlib.sha256(request['prompt'].encode()).hexdigest()!=item['prompt_sha256']:
                    raise ValueError('rollout rendered prompt differs')
                validate_sampling(params,2,17)
                name=digest(key);begin=time.monotonic()
                # Persist before dispatch; interrupted/ambiguous calls are never replayed.
                atomic_json(out/'requests'/f'{name}.intent.json',dict(example_id=key[0],decision_id=key[1],
                    policy_sha256=policy,sampling=effective,started_unix=time.time()))
                outputs=engine.generate([request],params,use_tqdm=False)
                atomic_json(out/'requests'/f'{name}.raw.json',dict(outputs=[dict(
                    finished=o.finished,prompt_token_ids=o.prompt_token_ids,outputs=[dict(
                        index=a.index,text=a.text,token_ids=list(a.token_ids),finish_reason=a.finish_reason,
                        logprobs=None if a.logprobs is None else [None if entry is None else
                            {str(t):v.logprob for t,v in entry.items()} for entry in a.logprobs])
                        for a in o.outputs]) for o in outputs]))
                if len(outputs)!=1:raise ValueError('one complete request output required')
                captured=capture(outputs[0],row,d['id'],item,processor.tokenizer,
                    policy_sha256=policy,instruction_sha256=identity['instruction_sha256'],group_size=2)
                atomic_json(out/'requests'/f'{name}.result.json',dict(records=captured,
                    input_ids=item['input_ids'].tolist(),seconds=time.monotonic()-begin))
                records.extend(captured)
                atomic_json(out/'progress.json',dict(phase='sampling',done=len(records),total=156,heartbeat_unix=time.time()))
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=30)
    verify_assets(base)
    if len(records)!=156:raise ValueError('complete 78 x 2 samples required')
    path=out/'records.jsonl';path.write_bytes(b''.join(map(canonical,records)))
    atomic_json(out/'result.json',dict(status='passed',bundles=6,decisions=78,responses=156,
        records_sha256=file_hash(path),identity_sha256=file_hash(out/'identity.json'),
        model_loaded=True,weight_updates=0,elapsed_seconds=time.monotonic()-start,
        scope='Frozen audio-direct K=2 generation with raw sampled-token logprobs; no HF forward parity, reward optimization or training'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:run(json.loads(a.config.read_text()),a.output,a.preflight_only)
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))

#!/usr/bin/env python3
"""Frozen complete TRAIN78 K2 AsyncLLM sampling; independent of queued sync run."""
import argparse
import asyncio
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.omni_actor_batch import prepare
from multimodal_jev.omni_async_rollout import OmniAsyncSampler
from multimodal_jev.sfc_audio import load_sfc_audio
from multimodal_jev.sfc_reward_batch import digest
from sample_sfc_omni_learning import load_inputs, run as sync_preflight
from infer_qwen3_omni_probe import verify_assets


def raw_output(output):
    if output is None:
        return None
    return dict(request_id=output.request_id, finished=output.finished,
        prompt_token_ids=output.prompt_token_ids, outputs=[dict(index=a.index, text=a.text,
            token_ids=list(a.token_ids), finish_reason=a.finish_reason,
            logprobs=None if a.logprobs is None else [None if values is None else
                {str(t):v.logprob for t,v in values.items()} for values in a.logprobs]) for a in output.outputs])


async def generate(engine_args, processor, rows, prepared, policy, identity, out):
    from vllm.v1.engine.async_llm import AsyncLLM
    engine=AsyncLLM.from_engine_args(engine_args)
    records=[]
    try:
        sampler=OmniAsyncSampler(engine, processor, policy_sha256=policy,
            instruction_sha256=identity['instruction_sha256'], max_model_len=4096)
        atomic_json(out/'effective-engine.json',dict(logprobs_mode=engine.model_config.logprobs_mode,
            generation_config=engine.model_config.generation_config,max_model_len=engine.model_config.max_model_len))
        for row in rows:
            waves=[]
            for path,pin in zip(row['input']['context']['audios'],row['audio_sha256'],strict=True):
                if file_hash(path)!=pin:raise ValueError('registered audio changed')
                waves.append(load_sfc_audio(path)[0])
            for decision in row['input']['decisions']:
                key=row['id'],decision['id'];name=digest(key);request_id=digest([policy,*key])
                intent=out/'requests'/f'{name}.intent.json'
                # Exclusive persistent admission before engine interaction. A new run
                # directory is required; no restart skips or repeats ambiguous requests.
                import os
                with intent.open('x') as handle:
                    json.dump(dict(request_id=request_id,example_id=key[0],decision_id=key[1],
                        policy_sha256=policy,group_size=2,seed=17,max_tokens=1024,
                        started_unix=time.time()),handle)
                    handle.flush();os.fsync(handle.fileno())
                begin=time.monotonic()
                try:
                    captured=await sampler.sample(row,decision['id'],prepared[key],waves,
                        request_id=request_id,group_size=2,seed=17,max_tokens=1024)
                except BaseException as exc:
                    atomic_json(out/'requests'/f'{name}.failure.json',dict(error=repr(exc),
                        notes=getattr(exc,'__notes__',[]),seconds=time.monotonic()-begin))
                    raise
                finally:
                    atomic_json(out/'requests'/f'{name}.raw.json',dict(output=raw_output(sampler.last_output)))
                atomic_json(out/'requests'/f'{name}.result.json',dict(records=captured,
                    input_ids=prepared[key]['input_ids'].tolist(),seconds=time.monotonic()-begin))
                records.extend(captured)
                atomic_json(out/'progress.json',dict(phase='sampling',done=len(records),total=156,
                                                    heartbeat_unix=time.time()))
    finally:
        engine.shutdown(timeout=30)
    if len(records)!=156:raise ValueError('all 78 decisions and K2 responses required')
    return records


def run(config,out,preflight_only):
    started=time.monotonic()
    (out/'preflight').mkdir()
    # Reuse fixed real processor/SamplingParams/learning-partition admission.
    sync_preflight(config,out/'preflight',True)
    identity=json.loads((out/'preflight/identity.json').read_text())
    from vllm.engine.arg_utils import AsyncEngineArgs
    args=AsyncEngineArgs(**identity['engine_arguments'])
    base,rows,_,_=load_inputs(config)
    identity.update(backend='vllm.v1.engine.async_llm.AsyncLLM',preflight_only=preflight_only,
        runner_sha256=file_hash(__file__),preflight_identity_sha256=file_hash(out/'preflight/identity.json'))
    atomic_json(out/'identity.json',identity)
    if preflight_only:
        atomic_json(out/'result.json',dict(status='passed',preflight_only=True,bundles=6,decisions=78,
            async_engine_args_constructed=True,model_loaded=False,weight_updates=0,
            identity_sha256=file_hash(out/'identity.json'),elapsed_seconds=time.monotonic()-started,
            scope='Real processor, learning input validation, SamplingParams and AsyncEngineArgs only; no engine or model'))
        return
    atomic_json(out/'progress.json',dict(phase='verify_weights',heartbeat_unix=time.time()))
    assets=verify_assets(base)
    policy=digest(dict(repository=assets['repository'],revision=assets['revision'],files=assets['files']))
    identity.update(policy_sha256=policy,model_files=assets['files'])
    atomic_json(out/'identity.json',identity)
    from transformers import Qwen3OmniMoeProcessor
    processor=Qwen3OmniMoeProcessor.from_pretrained(base['audio_model'],local_files_only=True)
    prepared=prepare(processor,rows)
    (out/'requests').mkdir()
    atomic_json(out/'progress.json',dict(phase='async_engine_load',heartbeat_unix=time.time()))
    records=asyncio.run(generate(args,processor,rows,prepared,policy,identity,out))
    verify_assets(base)
    path=out/'records.jsonl';path.write_bytes(b''.join(map(canonical,records)))
    atomic_json(out/'result.json',dict(status='passed',preflight_only=False,bundles=6,decisions=78,responses=156,
        records_sha256=file_hash(path),identity_sha256=file_hash(out/'identity.json'),
        model_loaded=True,weight_updates=0,elapsed_seconds=time.monotonic()-started,
        scope='Frozen direct audio AsyncLLM complete K2 groups; no sync parity, HF parity, Ray integration or training'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--config-hash',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    if file_hash(args.config)!=args.config_hash:raise ValueError('registered configuration changed')
    args.output.mkdir(parents=True,exist_ok=False)
    try:run(json.loads(args.config.read_text()),args.output,args.preflight_only)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
    atomic_json(args.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))

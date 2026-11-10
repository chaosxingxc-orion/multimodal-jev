#!/usr/bin/env python3
"""Generate every decision in a registered fixed-prompt SFC-Jev shard."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_audio import load_sfc_audio
from multimodal_jev.sfc_jev_runtime import vllm_request
from multimodal_jev.sfc_candidate_execution import execution_prompt
from multimodal_jev.sfc_prompt_candidates import vllm_request as candidate_request
from multimodal_jev.sfc_jev_shard_runtime import select_inputs,check_probe,check_predictions,digest
from infer_sfc_jev_omni_probe import accepted_response
from infer_qwen3_omni_probe import verify_assets
from preflight_sfc_jev_omni import expected_frames

SOURCE_FILES=(
    'tools/jev/infer_sfc_jev_omni_shard.py','tools/jev/infer_sfc_jev_omni_probe.py',
    'tools/jev/infer_qwen3_omni_probe.py','tools/jev/preflight_sfc_jev_omni.py',
    'src/multimodal_jev/sfc_jev_shard_runtime.py','src/multimodal_jev/sfc_jev_shards.py',
    'src/multimodal_jev/sfc_jev_runtime.py','src/multimodal_jev/sfc_jev.py',
    'src/multimodal_jev/sfc_definitions.py','src/multimodal_jev/sfc_consumption.py',
    'src/multimodal_jev/sfc.py','src/multimodal_jev/sfc_audio.py','src/multimodal_jev/sfc_omni.py',
    'src/multimodal_jev/sfc_partition.py','src/multimodal_jev/sfc_teacher_feedback.py',
    'src/multimodal_jev/sfc_native_full.py','src/multimodal_jev/consumption.py','src/multimodal_jev/baseline.py',
    'src/multimodal_jev/sfc_candidate_execution.py','src/multimodal_jev/sfc_candidate_context.py',
    'src/multimodal_jev/sfc_prompt_candidates.py','src/multimodal_jev/sfc_teacher_wire.py',
    'src/multimodal_jev/teacher_candidates.py',
)


def run(cfg,out):
    rows,plan,checks=select_inputs(cfg);probe=check_probe(cfg);start=time.monotonic()
    instruction,prompt_identity=execution_prompt(cfg)
    atomic_json(out/'progress.json',dict(phase='verifying_model',heartbeat_unix=time.time()))
    assets=verify_assets(cfg)
    versions={k:importlib.metadata.version(k) for k in ('torch','torch-npu','transformers','vllm','vllm-ascend')}
    if versions!=cfg['versions'] or assets['files']!=probe['model_files']:
        raise ValueError('same accepted TRAIN model and versions required')
    atomic_json(out/'identity.json',dict(config=cfg,model_files=assets['files'],versions=versions,
        inputs_sha256=digest(rows),context_result_sha256=cfg['context_result_sha256'],targets_read=False,
        source_files={n:file_hash(Path(__file__).parents[2]/n) for n in SOURCE_FILES}))
    keys=[[r['id'],d['id']] for r in rows for d in r['input']['decisions']]
    atomic_json(out/'selection.json',dict(ids=[r['id'] for r in rows],decisions=keys))
    from transformers import Qwen3OmniMoeProcessor
    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessorKwargs
    from vllm import LLM,SamplingParams
    import torch
    if Qwen3OmniMoeProcessorKwargs._defaults['audio_kwargs']['truncation'] is not False:
        raise ValueError('untruncated official processor required')
    processor=Qwen3OmniMoeProcessor.from_pretrained(cfg['audio_model'],local_files_only=True)
    atomic_json(out/'progress.json',dict(phase='engine_load',heartbeat_unix=time.time()))
    engine=LLM(model=cfg['audio_model'],dtype='bfloat16',tensor_parallel_size=2,seed=17,
        max_model_len=4096,max_num_seqs=1,max_num_batched_tokens=4096,gpu_memory_utilization=.85,
        enforce_eager=True,trust_remote_code=False,limit_mm_per_prompt={'audio':2,'image':0,'video':0},enable_prefix_caching=False)
    params=SamplingParams(temperature=0.,max_tokens=1024,n=1,seed=17)
    records=[];cache=out/'records';cache.mkdir()
    try:
        for row in rows:
            waves=[];conversions=[]
            for path,value in zip(row['input']['context']['audios'],row['audio_sha256'],strict=True):
                if file_hash(path)!=value:raise ValueError('audio changed')
                wave,conversion=load_sfc_audio(path);waves.append(wave);conversions.append(conversion)
            for d in row['input']['decisions']:
                key=(row['id'],d['id']);check=checks[key]
                request=(vllm_request(processor,row,d['id'],waves) if prompt_identity is None else
                         candidate_request(processor,row,d['id'],waves,instruction))
                batch=processor(text=[request['prompt']],audio=waves,sampling_rate=16000,return_tensors='pt',padding=True)
                ids=batch['input_ids'][0].tolist();prompt_hash=hashlib.sha256(request['prompt'].encode()).hexdigest()
                frames=expected_frames([len(w) for w in waves],processor.feature_extractor.hop_length)
                observed=batch['feature_attention_mask'].sum(-1).tolist()
                if (len(ids)+1024>4096 or len(ids)!=check['input_tokens'] or digest(ids)!=check['input_ids_sha256']
                        or prompt_hash!=check['prompt_sha256'] or frames!=check['expected_frames']
                        or observed!=frames or not torch.isfinite(batch['input_features']).all()):
                    raise ValueError('actual shard processor differs from accepted full context evidence')
                before=time.monotonic();responses=engine.generate([request],params,use_tqdm=False)
                if len(responses)!=1:raise ValueError('one generation required')
                answer=accepted_response(responses[0],ids)
                record=dict(example_id=row['id'],decision_id=d['id'],request_sha256=digest(row),
                    source_group=row['source_group'],split=row['split'],raw_text=answer.text,token_ids=list(answer.token_ids),
                    input_tokens=len(ids),hf_vllm_prompt_ids_equal=True,input_ids_sha256=digest(ids),
                    prompt_sha256=prompt_hash,audio_sha256=row['audio_sha256'],audio_conversion=conversions,
                    expected_frames=frames,observed_frames=observed,finish_reason=answer.finish_reason,
                    length_limited=answer.finish_reason=='length',seconds=time.monotonic()-before)
                atomic_json(cache/(digest(key)+'.json'),record);records.append(record)
                atomic_json(out/'progress.json',dict(phase='generation',done=len(records),total=len(keys),heartbeat_unix=time.time()))
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=30)
    atomic_json(out/'progress.json',dict(phase='final_model_verification',done=len(records),total=len(keys),heartbeat_unix=time.time()))
    verify_assets(cfg);check_predictions(rows,checks,records)
    if execution_prompt(cfg)!=(instruction,prompt_identity):raise ValueError('candidate changed during inference')
    path=out/'predictions.jsonl';path.write_bytes(b''.join(map(canonical,records)))
    atomic_json(out/'result.json',dict(status='passed',bundles=len(rows),decisions=len(keys),predictions=len(records),
        predictions_sha256=file_hash(path),identity_sha256=file_hash(out/'identity.json'),weight_updates=0,
        model_files_unchanged=True,targets_read=False,length_limited=sum(p['length_limited'] for p in records),
        elapsed_seconds=time.monotonic()-start,
        scope='One complete registered fixed-prompt shard. No targets or current transcripts read. '
              'All function decisions generated independently. Not a complete split score or interactive execution.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('config','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:run(json.loads(a.config.read_text()),a.output)
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))

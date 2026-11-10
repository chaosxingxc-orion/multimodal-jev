#!/usr/bin/env python3
"""Frozen audio-direct Jev generation for all 78 processor-validated TRAIN decisions."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_audio import load_sfc_audio
from multimodal_jev.sfc_jev_runtime import vllm_request
from infer_qwen3_omni_probe import verify_assets
from preflight_sfc_jev_omni import select, expected_frames

SOURCE_FILES = (
    'tools/jev/infer_sfc_jev_omni_probe.py', 'tools/jev/preflight_sfc_jev_omni.py',
    'tools/jev/infer_qwen3_omni_probe.py', 'src/multimodal_jev/sfc_jev_runtime.py',
    'src/multimodal_jev/sfc_jev.py', 'src/multimodal_jev/sfc_definitions.py',
    'src/multimodal_jev/sfc_consumption.py', 'src/multimodal_jev/sfc.py',
    'src/multimodal_jev/sfc_omni.py', 'src/multimodal_jev/sfc_audio.py',
    'src/multimodal_jev/consumption.py', 'src/multimodal_jev/baseline.py',
)


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def select_inputs(cfg):
    expected = dict(split='train_id', parameters_frozen=True, route='audio-direct',
                    dtype='bfloat16', seed=17, max_context_tokens=4096, max_new_tokens=1024,
                    sampling={'temperature':0., 'n':1})
    if any(cfg.get(k) != v for k,v in expected.items()) or cfg.get('adapter') is not None:
        raise ValueError('registered frozen TRAIN audio-direct runtime required')
    root = Path(cfg['processor_preflight'])/'run'
    result = json.loads((root/'result.json').read_text())
    identity = json.loads((root/'identity.json').read_text())
    if (file_hash(root/'result.json') != cfg['processor_preflight_sha256']
            or file_hash(root/'identity.json') != result['identity_sha256']
            or json.loads((root/'terminal.json').read_text())['exit_code'] != 0
            or result['status'] != 'passed' or identity['model'] != cfg['audio_model']
            or result['decisions'] != 78 or result['bundles'] != 6):
        raise ValueError('successful pinned 78-decision processor preflight required')
    for name, value in identity['source_files'].items():
        if file_hash(Path(__file__).parents[2]/name) != value:
            raise ValueError('processor implementation changed')
    release = Path(identity['release'])
    if file_hash(release/'MANIFEST.json') != identity['manifest_sha256']:
        raise ValueError('release manifest changed')
    path = release/'train_id.requests.jsonl'
    if file_hash(path) != identity['train_requests_sha256']:
        raise ValueError('TRAIN requests changed')
    rows = select([json.loads(line) for line in path.open()])
    keys = [(r['id'], d['id']) for r in rows for d in r['input']['decisions']]
    if ([r['id'] for r in rows] != identity['selected_ids'] or len(keys) != 78
            or [(r['id'], r['decision_id']) for r in result['records']] != keys
            or any(r['input_tokens']+1024 > 4096 or r['expected_frames'] != r['observed_frames']
                   for r in result['records'])):
        raise ValueError('exact complete preflight decision population required')
    return rows, identity, {(r['id'],r['decision_id']):r for r in result['records']}


def accepted_response(response, expected_ids):
    if not response.finished or len(response.outputs) != 1:
        raise ValueError('one complete response required')
    if list(response.prompt_token_ids) != expected_ids:
        raise ValueError('HF and vLLM expanded prompt IDs differ')
    answer = response.outputs[0]
    if (answer.finish_reason not in ('stop','length') or len(answer.token_ids) > 1024
            or type(answer.text) is not str):
        raise ValueError('unexpected generation termination')
    return answer


def run(cfg, out):
    rows, preflight, checks = select_inputs(cfg); start=time.monotonic()
    atomic_json(out/'progress.json',dict(phase='verifying_model',heartbeat_unix=time.time()))
    assets=verify_assets(cfg)
    versions={k:importlib.metadata.version(k) for k in ('torch','torch-npu','transformers','vllm','vllm-ascend')}
    if versions != cfg['versions']:
        raise ValueError('pinned inference versions required')
    for name, value in preflight['processor_assets'].items():
        if file_hash(Path(cfg['audio_model'])/name)!=value:raise ValueError('processor asset changed')
    identity=dict(config=cfg,model_files=assets['files'],versions=versions,inputs_sha256=digest(rows),
                  processor_preflight_sha256=cfg['processor_preflight_sha256'],
                  source_files={n:file_hash(Path(__file__).parents[2]/n) for n in SOURCE_FILES})
    atomic_json(out/'identity.json',identity)
    atomic_json(out/'selection.json',dict(ids=[r['id'] for r in rows],decisions=[list(k) for k in checks]))
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams
    import torch
    processor=Qwen3OmniMoeProcessor.from_pretrained(cfg['audio_model'],local_files_only=True)
    atomic_json(out/'progress.json',dict(phase='engine_load',heartbeat_unix=time.time()))
    engine=LLM(model=cfg['audio_model'],dtype='bfloat16',tensor_parallel_size=2,seed=17,
        max_model_len=4096,max_num_seqs=1,max_num_batched_tokens=4096,
        gpu_memory_utilization=.85,enforce_eager=True,trust_remote_code=False,
        limit_mm_per_prompt={'audio':2,'image':0,'video':0},enable_prefix_caching=False)
    params=SamplingParams(temperature=0.,max_tokens=1024,n=1,seed=17)
    records=[];cache=out/'records';cache.mkdir()
    try:
        for row in rows:
            waves=[];conversions=[]
            for path,value in zip(row['input']['context']['audios'],row['audio_sha256'],strict=True):
                if file_hash(path)!=value:raise ValueError('audio changed')
                wave,conversion=load_sfc_audio(path);waves.append(wave);conversions.append(conversion)
            for decision in row['input']['decisions']:
                key=(row['id'],decision['id']);check=checks[key]
                request=vllm_request(processor,row,decision['id'],waves)
                batch=processor(text=[request['prompt']],audio=waves,sampling_rate=16000,return_tensors='pt',padding=True)
                expected_ids=batch['input_ids'][0].tolist()
                prompt_hash=hashlib.sha256(request['prompt'].encode()).hexdigest()
                frames=expected_frames([len(w) for w in waves],processor.feature_extractor.hop_length)
                if (len(expected_ids)+1024>4096 or digest(expected_ids)!=check['input_ids_sha256']
                        or prompt_hash!=check['prompt_sha256'] or frames!=check['expected_frames']
                        or batch['feature_attention_mask'].sum(-1).tolist()!=frames
                        or not torch.isfinite(batch['input_features']).all()):
                    raise ValueError('processor parity, feature coverage or context changed')
                before=time.monotonic();responses=engine.generate([request],params,use_tqdm=False)
                if len(responses)!=1:raise ValueError('one response required')
                answer=accepted_response(responses[0],expected_ids)
                record=dict(example_id=row['id'],decision_id=decision['id'],request_sha256=digest(row),
                    source_group=row['source_group'],split='train_id',raw_text=answer.text,token_ids=list(answer.token_ids),
                    input_tokens=len(expected_ids),hf_vllm_prompt_ids_equal=True,input_ids_sha256=digest(expected_ids),
                    prompt_sha256=prompt_hash,audio_sha256=row['audio_sha256'],audio_conversion=conversions,
                    expected_frames=frames,finish_reason=answer.finish_reason,length_limited=answer.finish_reason=='length',
                    seconds=time.monotonic()-before)
                atomic_json(cache/(digest(key)+'.json'),record);records.append(record)
                atomic_json(out/'progress.json',dict(phase='generation',done=len(records),total=78,heartbeat_unix=time.time()))
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=30)
    atomic_json(out/'progress.json',dict(phase='final_model_verification',done=len(records),total=78,heartbeat_unix=time.time()))
    verify_assets(cfg)
    if len(records)!=78:raise ValueError('incomplete generation')
    path=out/'predictions.jsonl';path.write_bytes(b''.join(map(canonical,records)))
    atomic_json(out/'result.json',dict(status='passed',bundles=6,decisions=78,predictions=78,
        predictions_sha256=file_hash(path),weight_updates=0,model_files_unchanged=True,
        length_limited=sum(r['length_limited'] for r in records),elapsed_seconds=time.monotonic()-start,
        scope='All 78 decisions in six TRAIN turns. Engineering acceptance independent of output quality. '
              'No weight updates, TEST inference or causal audio-use claim.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('config','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:run(json.loads(a.config.read_text()),a.output)
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))

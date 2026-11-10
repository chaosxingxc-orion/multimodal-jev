#!/usr/bin/env python3
"""Frozen native SFC inference for a complete official TEST split; no target reads."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_audio import load_sfc_audio
from multimodal_jev.sfc_omni import vllm_request
from multimodal_jev.sfc_native_full import select_inputs, check_probe, check_predictions, digest
from infer_qwen3_omni_probe import verify_assets
from check_sfc_omni_long_audio import expected_mask_frames

SOURCE_FILES = (
    'tools/jev/infer_sfc_omni_full.py', 'tools/jev/infer_qwen3_omni_probe.py',
    'tools/jev/check_sfc_omni_long_audio.py', 'src/multimodal_jev/sfc_native_full.py',
    'src/multimodal_jev/sfc_omni.py', 'src/multimodal_jev/sfc_audio.py',
    'src/multimodal_jev/sfc_consumption.py', 'src/multimodal_jev/sfc.py',
    'src/multimodal_jev/baseline.py', 'src/multimodal_jev/consumption.py',
)


def accept_response(responses, expected_ids):
    if len(responses) != 1 or not responses[0].finished or len(responses[0].outputs) != 1:
        raise ValueError('one complete generation required')
    response = responses[0]; answer = response.outputs[0]
    if list(response.prompt_token_ids) != expected_ids:
        raise ValueError('HF and vLLM expanded prompt IDs differ')
    if (answer.finish_reason not in ('stop', 'length') or len(answer.token_ids) > 1024
            or type(answer.text) is not str):
        raise ValueError('unexpected generation termination')
    return answer


def run(cfg, out):
    rows = select_inputs(cfg); probe = check_probe(cfg); start = time.monotonic()
    atomic_json(out/'progress.json', dict(phase='verifying_model', heartbeat_unix=time.time()))
    assets = verify_assets(cfg)
    versions = {k: importlib.metadata.version(k) for k in ('torch','torch-npu','transformers','vllm','vllm-ascend')}
    if versions != cfg['versions'] or assets['files'] != probe['model_files']:
        raise ValueError('same pinned TRAIN model and inference versions required')
    identity = dict(config=cfg, model_files=assets['files'], versions=versions, inputs_sha256=digest(rows),
        native_probe_acceptance_sha256=cfg['native_probe_acceptance_sha256'], targets_read=False,
        source_files={n: file_hash(Path(__file__).parents[2]/n) for n in SOURCE_FILES})
    atomic_json(out/'identity.json', identity)
    atomic_json(out/'selection.json', dict(ids=[r['id'] for r in rows]))
    from transformers import Qwen3OmniMoeProcessor
    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessorKwargs
    from vllm import LLM, SamplingParams
    import torch
    if Qwen3OmniMoeProcessorKwargs._defaults['audio_kwargs']['truncation'] is not False:
        raise ValueError('untruncated official audio processing required')
    processor = Qwen3OmniMoeProcessor.from_pretrained(cfg['audio_model'], local_files_only=True)
    atomic_json(out/'progress.json', dict(phase='engine_load', heartbeat_unix=time.time()))
    engine = LLM(model=cfg['audio_model'], dtype='bfloat16', tensor_parallel_size=2, seed=17,
        max_model_len=4096, max_num_seqs=1, max_num_batched_tokens=4096,
        gpu_memory_utilization=.85, enforce_eager=True, trust_remote_code=False,
        limit_mm_per_prompt={'audio':2,'image':0,'video':0}, enable_prefix_caching=False)
    params = SamplingParams(temperature=0., max_tokens=1024, n=1, seed=17)
    records = []; cache = out/'records'; cache.mkdir()
    try:
        for row in rows:
            waves = []; conversions = []
            for path, value in zip(row['input']['audios'], row['audio_sha256'], strict=True):
                if file_hash(path) != value:
                    raise ValueError('audio changed')
                wave, conversion = load_sfc_audio(path)
                waves.append(wave); conversions.append(conversion)
            request = vllm_request(processor, row['input'], waves)
            batch = processor(text=[request['prompt']], audio=waves, sampling_rate=16000, return_tensors='pt', padding=True)
            expected_ids = batch['input_ids'][0].tolist()
            frames = expected_mask_frames([len(w) for w in waves], processor.feature_extractor.hop_length)
            observed = batch['feature_attention_mask'].sum(-1).tolist()
            if (len(expected_ids)+1024 > 4096 or frames != observed
                    or not torch.isfinite(batch['input_features']).all()):
                raise ValueError('complete audio coverage or context budget failed')
            before = time.monotonic()
            answer = accept_response(engine.generate([request], params, use_tqdm=False), expected_ids)
            record = dict(example_id=row['id'], source_group=row['source_group'], split=row['split'],
                request_sha256=digest(row), raw_text=answer.text, token_ids=list(answer.token_ids),
                input_tokens=len(expected_ids), input_ids_sha256=digest(expected_ids), hf_vllm_prompt_ids_equal=True,
                audio_sha256=row['audio_sha256'], audio_conversion=conversions,
                prompt_sha256=hashlib.sha256(request['prompt'].encode()).hexdigest(),
                expected_frames=frames, observed_frames=observed, finish_reason=answer.finish_reason,
                length_limited=answer.finish_reason=='length', seconds=time.monotonic()-before)
            check_predictions([row], [record])
            atomic_json(cache/(digest(row['id'])+'.json'), record); records.append(record)
            atomic_json(out/'progress.json',dict(phase='generation',done=len(records),total=len(rows),heartbeat_unix=time.time()))
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=30)
    atomic_json(out/'progress.json',dict(phase='final_model_verification',done=len(records),total=len(rows),heartbeat_unix=time.time()))
    verify_assets(cfg); check_predictions(rows, records)
    path = out/'predictions.jsonl'; path.write_bytes(b''.join(map(canonical, records)))
    atomic_json(out/'result.json', dict(status='passed', requests=len(rows), predictions=len(records),
        predictions_sha256=file_hash(path), identity_sha256=file_hash(out/'identity.json'),
        weight_updates=0, model_files_unchanged=True, targets_read=False,
        length_limited=sum(p['length_limited'] for p in records), elapsed_seconds=time.monotonic()-start,
        scope='Complete official native SFC split with supplied history. No current target reads, '
              'no transcript intermediate, no weight updates; not interactive tool execution.'))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for k in ('config','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:run(json.loads(a.config.read_text()),a.output)
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))

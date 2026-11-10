#!/usr/bin/env python3
"""Full 1000-input MMAR CPU HF/vLLM processing; no weights or answers."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.mmar_omni import INPUT_SHA, INSTRUCTION, prepare_request


def run(args):
    import torch
    from transformers import Qwen3OmniMoeProcessor
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.processing.inputs import ProcessorInputs
    from vllm.multimodal.processing.context import TimingContext
    torch.set_num_threads(4)
    start = time.monotonic()
    if file_hash(args.inputs) != INPUT_SHA:
        raise ValueError('registered full input-only MMAR release required')
    rows = [json.loads(line) for line in args.inputs.open()]
    if len(rows) != 1000 or len({r['id'] for r in rows}) != 1000:
        raise ValueError('all 1000 original questions required')
    assets = json.loads(args.assets.read_text())
    for name, pin in assets.items():
        if file_hash(args.model/name) != pin:
            raise ValueError('processor asset changed')
    hf = Qwen3OmniMoeProcessor.from_pretrained(args.model, local_files_only=True)
    config = ModelConfig(model=str(args.model), tokenizer=str(args.model), dtype='bfloat16',
                         max_model_len=4096, enforce_eager=True,
                         limit_mm_per_prompt={'audio': 1, 'image': 0, 'video': 0})
    processor = MULTIMODAL_REGISTRY.create_processor(config, tokenizer=hf.tokenizer)
    versions = {k: importlib.metadata.version(k) for k in ('torch', 'torch-npu', 'transformers', 'vllm', 'vllm-ascend')}
    atomic_json(args.output/'identity.json', dict(inputs_sha256=INPUT_SHA, processor_assets=assets,
        versions=versions, instruction=INSTRUCTION, max_model_len=4096, reserved_output_tokens=256,
        targets_read=False, model_loaded=False))
    records = []
    with (args.output/'records.jsonl').open('x') as stream:
        for index, row in enumerate(rows):
            atomic_json(args.output/'progress.json', dict(phase='processing', done=index, total=1000, heartbeat_unix=time.time()))
            request, receipt = prepare_request(hf, row, args.audio_root)
            wave = request['multi_modal_data']['audio'][0][0]
            parsed = processor.info.get_data_parser().parse_mm_data(request['multi_modal_data'])
            processed = processor.apply(ProcessorInputs(
                prompt=hf.tokenizer.encode(request['prompt'], add_special_tokens=False),
                mm_data_items=parsed), TimingContext(enabled=False))
            data = processed['mm_kwargs'].get_data()
            feature = data['input_audio_features']
            batch = hf(text=[request['prompt']], audio=[wave], sampling_rate=16000,
                       return_tensors='pt', padding=True, truncation=False)
            mask = batch['feature_attention_mask']
            reference = batch['input_features'].permute(0, 2, 1)[mask.bool()].permute(1, 0).to(feature.dtype)
            ids = processed['prompt_token_ids']
            frames = data['audio_feature_lengths'].tolist()
            if (frames != [receipt['expected_frames']] or mask.sum(-1).tolist() != frames
                    or ids != batch['input_ids'][0].tolist() or len(ids)+256 > 4096
                    or feature.dtype != torch.bfloat16 or not torch.isfinite(feature).all()
                    or not torch.equal(feature, reference)):
                raise ValueError('full-frame exact HF/vLLM processor parity or context budget failed')
            record = dict(id=row['id'], audio=receipt, input_tokens=len(ids), frames=frames,
                prompt_sha256=hashlib.sha256(request['prompt'].encode()).hexdigest(),
                input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                input_ids_equal=True, bf16_features_equal=True)
            stream.write(json.dumps(record)+'\n'); stream.flush()
            records.append(record)
    atomic_json(args.output/'result.json', dict(status='passed', records=len(records),
        max_input_tokens=max(r['input_tokens'] for r in records),
        long_audio_count=sum(r['audio']['conversion']['target_samples'] > 30*16000 for r in records),
        records_sha256=file_hash(args.output/'records.jsonl'),
        identity_sha256=file_hash(args.output/'identity.json'), elapsed_seconds=time.monotonic()-start,
        targets_read=False, model_loaded=False, weight_updates=0,
        scope='All 1000 official inputs, real CPU HF/vLLM processor parity and full frames; no engine generation or quality result'))
    atomic_json(args.output/'progress.json', dict(phase='complete', done=1000, total=1000, heartbeat_unix=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('inputs', 'audio-root', 'model', 'assets', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json', dict(exit_code=1, error=repr(exc))); raise
    atomic_json(args.output/'terminal.json', dict(exit_code=0))

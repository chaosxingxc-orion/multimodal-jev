#!/usr/bin/env python3
"""Every scheduled real waveform/view through the actual Qwen ASR processor.

CPU-only input validation. Does not load model weights or admit backend parity.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.slurp_official_training import load, example


def run(config, output):
    import torch
    import soundfile as sf
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.span_contract import encode_span_request, span_labels
    from multimodal_jev.span_request_identity import tensor_identity
    if list(Path('/dev').glob('davinci[0-9]*')):
        raise ValueError('CPU-only processor validation')
    torch.set_num_threads(4)
    for path, digest in config['pins'].items():
        if file_hash(path) != digest:
            raise ValueError('preflight dependency changed: '+path)
    schedule_root = Path(config['schedule'])
    if file_hash(schedule_root/'MANIFEST.json') != config['schedule_sha256']:
        raise ValueError('fixed full schedule required')
    schedule = json.loads((schedule_root/'MANIFEST.json').read_text())
    data, task, release = load(schedule['release'], schedule['manifest_sha256'], 'learning', purpose='optimize')
    by_id = {r['id']: r for r in data}
    processor = Qwen3ASRProcessor.from_pretrained(config['model'], local_files_only=True)
    counts = Counter(); lengths = []; failures = []; unique_views = {}; total = 0; started = time.time()
    # Emit incrementally, no repeated copies of full input features or prompts.
    with (output/'records.jsonl').open('w') as records:
        for epoch in schedule['epochs']:
            p = schedule_root/f'epoch-{epoch["epoch"]}.json'
            if file_hash(p) != epoch['file_sha256']:
                raise ValueError('epoch file changed')
            batches = json.loads(p.read_text())['batches']
            if content_hash(batches) != epoch['batches_sha256']:
                raise ValueError('epoch contents changed')
            for batch_index, batch in enumerate(batches):
                for index, slot in enumerate(batch):
                    if slot['padding']:
                        continue
                    row = by_id[slot['id']]; total += 1
                    key = (slot['id'], slot['view'], slot['variant'])
                    if key in unique_views:
                        result = dict(unique_views[key], reused=True)
                    else:
                        if file_hash(row['audio']) != row['audio_sha256']:
                            raise ValueError('audio changed after publication')
                        audio, rate = sf.read(row['audio'], dtype='float32')
                        if rate != 16000 or audio.ndim != 1:
                            raise ValueError('unmodified mono16k waveform required')
                        request, target, binding = example(row, task, slot['view'], slot['variant'])
                        inputs, positions = encode_span_request(processor, request, audio)
                        mapping = span_labels(processor.tokenizer, request.questions)
                        length = inputs['input_ids'].shape[-1]
                        if length > config['context_limit']:
                            failures.append(dict(id=row['id'], view=slot['view'], variant=slot['variant'], input_tokens=length))
                        result = dict(id=row['id'], view=slot['view'], variant=slot['variant'],
                            input_tokens=length, positions=positions.tolist(),
                            candidate_counts=[len(q.values) for q in request.questions],
                            input_sha256=content_hash(request.model_input()),
                            input_ids=tensor_identity(inputs['input_ids']),
                            audio_sha256=row['audio_sha256'], samples=len(audio))
                        unique_views[key] = result
                    counts[slot['view']] += 1; lengths.append(result['input_tokens'])
                    records.write(json.dumps(dict(epoch=epoch['epoch'], batch=batch_index, slot=index, **result))+'\n')
                atomic_json(output/'progress.json', dict(completed=total, total=schedule['total_nonpadding_recording_uses'],
                    epoch=epoch['epoch'], batch=batch_index, maximum_tokens=max(lengths), updated_unix=time.time()))
    if total != schedule['total_nonpadding_recording_uses']:
        raise ValueError('full schedule not covered')
    return dict(passed=not failures, requests=total, unique_input_views=len(unique_views), views=dict(counts),
        maximum_tokens=max(lengths), minimum_tokens=min(lengths), context_limit=config['context_limit'],
        context_overflow=failures, seconds=time.time()-started, records_sha256=file_hash(output/'records.jsonl'),
        schedule_sha256=config['schedule_sha256'], release_sha256=schedule['manifest_sha256'],
        scope='Every scheduled nonpadding audio/input through real CPU Qwen3ASR processor; no model weights, NPU, training or HF/vLLM probability acceptance.')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    try:
        result=run(json.loads(a.config.read_text()),a.output);atomic_json(a.output/'result.json',result)
        if not result['passed']:
            raise ValueError('context overflow; no silent truncation or sample dropping')
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise

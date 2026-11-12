#!/usr/bin/env python3
"""Compile complete first two scheduled batches as an independent parity set.

Every one of the 256 training slots is represented; no model-score selection.
No targets or supervision enter the input artifact.
"""
import argparse
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.slurp_training import content_hash


def run(config, output):
    import torch
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.slurp_official_training import load, example
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.span_contract import encode_span_request, span_labels
    from multimodal_jev.span_request_identity import tensor_identity
    if list(Path('/dev').glob('davinci[0-9]*')):
        raise ValueError('CPU-only independent input preparation')
    torch.set_num_threads(4)
    for path, digest in config['pins'].items():
        if file_hash(path) != digest:
            raise ValueError('input dependency changed')
    rows, task, release = load(config['release'], config['release_sha256'], 'learning', purpose='optimize')
    population = {r['id']: r for r in rows}
    schedule = json.loads(Path(config['epoch_file']).read_text())
    if (schedule['report']['global_batch'] != 128 or schedule['report']['epoch'] != 0
            or content_hash(schedule['batches']) != schedule['report']['batches_sha256']):
        raise ValueError('full-data epoch0 global128 schedule required')
    processor = Qwen3ASRProcessor.from_pretrained(config['model'], local_files_only=True)
    records = []
    for step, batch in enumerate(schedule['batches'][:2]):
        for index, slot in enumerate(batch):
            if slot['padding'] or slot['weight'] != 1.:
                raise ValueError('first two complete source batches required')
            row = population[slot['id']]
            if file_hash(row['audio']) != row['audio_sha256']:
                raise ValueError('audio changed')
            request, _, _ = example(row, task, slot['view'], slot['variant'])
            audio, receipt = load_audio(row['audio'], return_receipt=True)
            inputs, positions = encode_span_request(processor, request, audio)
            length = inputs['input_ids'].shape[-1]
            if length > 4096:
                raise ValueError('input overflow; never truncate')
            uid = f'{row["id"]}::epoch0-batch{step}-slot{index}'
            records.append(dict(id=uid, recording_id=row['id'], batch=step, slot=index,
                view=slot['view'], variant=slot['variant'], request=request.model_input(),
                audio_sha256=row['audio_sha256'], audio_processing=receipt, input_tokens=length,
                positions=positions.tolist(), candidate_mapping=span_labels(processor.tokenizer, request.questions),
                **{k: tensor_identity(inputs[k]) for k in ('input_ids','input_features','input_features_mask')}))
        atomic_json(output/'progress.json', dict(completed=len(records), total=256, updated_unix=time.time()))
    if len(records) != 256 or len({r['recording_id'] for r in records}) != 256:
        raise ValueError('all 256 distinct scheduled sources required')
    return dict(passed=True, recordings=256, requests=256, records=records,
        release_sha256=config['release_sha256'], epoch_sha256=file_hash(config['epoch_file']),
        maximum_input_tokens=max(r['input_tokens'] for r in records),
        scope='Actual processor on all first two full-schedule batches; independent label-free parity population, not an efficacy subset.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    try:
        result=run(json.loads(a.config.read_text()),a.output);atomic_json(a.output/'result.json',result)
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise

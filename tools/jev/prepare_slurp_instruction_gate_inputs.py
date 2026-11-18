#!/usr/bin/env python3
"""Compile complete first two scheduled batches as an independent parity set.

Every one of the 256 training slots is represented; no model-score selection.
No targets or supervision enter the input artifact.
"""
import argparse
import json
import math
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
    selection = config.get('population_mode') == 'full_selection'
    if config.get('population_mode', 'prefix') not in ('prefix', 'full_selection'):
        raise ValueError('known input population required')
    rows, task, release = load(config['release'], config['release_sha256'],
        'selection' if selection else 'learning', purpose='evaluate' if selection else 'optimize')
    population = {r['id']: r for r in rows}
    if selection:
        if config.get('task_mode') != 'direct_final':
            raise ValueError('full selection requires direct native Final')
        schedule = None
        batches = [[dict(id=row['id'], padding=False, weight=1., view='direct', variant='native')]
                   for row in rows]
        expected_count = len(rows)
    else:
        schedule = json.loads(Path(config['epoch_file']).read_text())
        if (schedule['report']['global_batch'] != 128 or schedule['report']['epoch'] != 0
            or content_hash(schedule['batches']) != schedule['report']['batches_sha256']):
            raise ValueError('full-data epoch0 global128 schedule required')
        batches = schedule['batches'][:2]
        expected_count = 256
    processor = Qwen3ASRProcessor.from_pretrained(config['model'], local_files_only=True)
    records = []
    for step, batch in enumerate(batches):
        for index, slot in enumerate(batch):
            if (slot['padding'] or type(slot['weight']) not in (int, float)
                    or not math.isfinite(slot['weight']) or slot['weight'] <= 0
                    or (config.get('task_mode') != 'direct_final' and slot['weight'] != 1.)):
                raise ValueError('first two complete source batches required')
            if config.get('task_mode') == 'direct_final' and (slot['view'],slot['variant']) != ('direct','native'):
                raise ValueError('simple baseline input changed')
            row = population[slot['id']]
            if file_hash(row['audio']) != row['audio_sha256']:
                raise ValueError('audio changed')
            if selection:
                from multimodal_jev.slurp_intent_structure import compile_intent
                request = compile_intent(row['audio'], task, 'D')
            else:
                request, _, _ = example(row, task, slot['view'], slot['variant'])
            audio, receipt = load_audio(row['audio'], return_receipt=True)
            inputs, positions = encode_span_request(processor, request, audio)
            length = inputs['input_ids'].shape[-1]
            if length > 4096:
                raise ValueError('input overflow; never truncate')
            uid = row['id'] if selection else f'{row["id"]}::epoch0-batch{step}-slot{index}'
            records.append(dict(id=uid, recording_id=row['id'], batch=step, slot=index,
                view=slot['view'], variant=slot['variant'], request=request.model_input(),
                audio_sha256=row['audio_sha256'], audio_processing=receipt, input_tokens=length,
                positions=positions.tolist(), candidate_mapping=span_labels(processor.tokenizer, request.questions),
                **{k: tensor_identity(inputs[k]) for k in ('input_ids','input_features','input_features_mask')}))
        if not selection or (step+1) % 128 == 0:
            atomic_json(output/'progress.json', dict(completed=len(records), total=expected_count, updated_unix=time.time()))
    if len(records) != expected_count or len({r['recording_id'] for r in records}) != expected_count:
        raise ValueError('all distinct population sources required')
    return dict(passed=True, recordings=expected_count, requests=expected_count, records=records,
        population_mode='full_selection' if selection else 'prefix',
        release_sha256=config['release_sha256'], epoch_sha256=None if selection else file_hash(config['epoch_file']),
        maximum_input_tokens=max(r['input_tokens'] for r in records),
        scope='Actual processor on complete declared population; label-free input artifact, no model execution or task metrics.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    try:
        result=run(json.loads(a.config.read_text()),a.output);atomic_json(a.output/'result.json',result)
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise

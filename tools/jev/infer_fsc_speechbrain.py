#!/usr/bin/env python3
"""Pinned official SpeechBrain decode_batch on complete FSC splits, resumable."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.fsc_published import score_predictions


def run(config, output):
    import torch
    import numpy as np
    import soundfile as sf
    from speechbrain.inference.SLU import EndToEndSLU
    torch.set_num_threads(config['cpu_threads']); torch.manual_seed(config['seed'])
    root = Path(config['dataset'])
    csv_path = root / 'data' / (config['split'] + '_data.csv')
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    if config.get('example_ids') is not None:
        wanted = config['example_ids']
        if config['split'] != 'train' or len(wanted) != len(set(wanted)):
            raise ValueError('explicit selection is TRAIN-only and unique')
        rows = [row for row in rows if row['path'] in wanted]
        if len(rows) != len(wanted):
            raise ValueError('selection not found')
    if len(rows) != config['expected_records']:
        raise ValueError('unexpected split count')
    identity = {'config': config, 'csv_sha256': file_hash(csv_path),
                'runner_sha256': file_hash(__file__),
                'scorer_sha256': file_hash(Path(__file__).parents[2]/'src/multimodal_jev/fsc_published.py'),
                'models': {}}
    for name in ('slu_model', 'asr_model'):
        model_root = Path(config[name])
        verification = json.loads((model_root/'VERIFICATION.json').read_text())
        for entry in verification['files']:
            if file_hash(model_root/entry['file']) != entry['sha256']:
                raise ValueError('model file changed')
        identity['models'][name] = verification
    if (output/'identity.json').exists() and json.loads((output/'identity.json').read_text()) != identity:
        raise ValueError('resume identity mismatch')
    atomic_json(output/'identity.json', identity)
    atomic_json(output/'progress.json', {'phase': 'load', 'time': time.time()})
    model = EndToEndSLU.from_hparams(source=config['slu_model'], savedir=str(output/'cache'),
        overrides={'asr_model_source': config['asr_model']}, run_opts={'device': config['device']})
    assert model.hparams.slu_beam_size == 80 and model.hparams.temperature == 1.25
    records = []; cache = output/'records'; cache.mkdir(exist_ok=True)
    start = time.monotonic()
    for index, row in enumerate(rows):
        # Labels and transcripts are never supplied to decode_batch.
        audio = (root/row['path']).resolve(); audio.relative_to(root.resolve())
        digest = file_hash(audio); path = cache/f'{index:05d}.json'
        if path.exists():
            record = json.loads(path.read_text())
            if record['example_id'] != row['path'] or record['audio_sha256'] != digest:
                raise ValueError('cached prediction identity mismatch')
        else:
            wave, rate = sf.read(audio, dtype='float32')
            if rate != 16000 or wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
                raise ValueError('invalid mono16k audio')
            before = time.monotonic()
            raw, tokens = model.decode_batch(torch.from_numpy(wave)[None], torch.ones(1))
            if len(raw) != 1 or len(tokens) != 1:
                raise ValueError('incomplete model prediction')
            record = {'example_id': row['path'], 'raw_prediction': raw[0],
                      'token_ids': tokens[0], 'audio_sha256': digest,
                      'decode_seconds': time.monotonic()-before}
            atomic_json(path, record)
        records.append(record)
        atomic_json(output/'progress.json', {'phase': 'inference', 'done': index+1,
            'total': len(rows), 'elapsed_seconds': time.monotonic()-start, 'heartbeat_unix': time.time()})
        if sum(record['decode_seconds'] for record in records) > config['max_decode_seconds']:
            raise RuntimeError('registered decode time budget exhausted; partial outputs retained')
    predictions = output/'predictions.jsonl'
    predictions.write_text(''.join(json.dumps(row)+'\n' for row in records))
    result = score_predictions(rows, records)
    result.update(status='passed', split=config['split'], predictions_sha256=file_hash(predictions),
                  elapsed_seconds=time.monotonic()-start,
                  comparison='published weights on current pinned official inference; historical training environment not reproduced')
    atomic_json(output/'result.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output/'result.json').exists():
            raise FileExistsError('completed inference')
        try:
            run(json.loads(args.config.read_text()), args.output)
        except BaseException as exc:
            atomic_json(args.output/'terminal.json', {'exit_code': 1, 'error': repr(exc), 'time': time.time()})
            raise
        atomic_json(args.output/'terminal.json', {'exit_code': 0, 'time': time.time()})

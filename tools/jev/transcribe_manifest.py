#!/usr/bin/env python3
"""Resumable Qwen3-ASR transcription baseline on a fixed Jev manifest.

This measures ASR only. References stay in metadata and are never passed to the
model; the result is not used for model selection.
"""
import argparse
import fcntl
import hashlib
import json
import re
import time
from pathlib import Path
from multimodal_jev.baseline import atomic_json, file_hash


def normalize(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, token in enumerate(reference, 1):
        current = [i]
        for j, value in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (token != value)))
        previous = current
    return previous[-1]


def run(config, output):
    import torch
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.audio_policy import load_audio
    output = Path(output); manifest = Path(config['manifest'])
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if not rows or len({r['example_id'] for r in rows}) != len(rows):
        raise ValueError('manifest must contain unique nonempty examples')
    if any(r['metadata'].get('official_split', r['metadata'].get('partition', 'valid')) != config['split']
           for r in rows):
        raise ValueError('manifest split mismatch')
    identity = {'config': config, 'manifest_sha256': file_hash(manifest),
                'model_sha256': file_hash(Path(config['model_path'])/'model.safetensors'),
                'runner_sha256': file_hash(__file__)}
    if (output/'identity.json').exists() and json.loads((output/'identity.json').read_text()) != identity:
        raise ValueError('resume identity mismatch')
    atomic_json(output/'identity.json', identity)
    processor, model = load_runtime(config['model_path'], config['device'], config['dtype'])
    model.eval(); cache = output/'records'; cache.mkdir(exist_ok=True); start = time.monotonic()
    representatives = {}
    for row in rows:
        representatives.setdefault(row['metadata']['audio_sha256'], row)
    audio_results = {}
    for index, row in enumerate(representatives.values()):
        audio = Path(row['audio']); audio_key = row['metadata']['audio_sha256']; key = row['example_id']
        if file_hash(audio) != audio_key:
            raise ValueError('audio changed')
        path = cache/(hashlib.sha256(('audio:'+audio_key).encode()).hexdigest()+'.json')
        if path.exists():
            record = json.loads(path.read_text())
            if record['audio_sha256'] != audio_key: raise ValueError('cache audio ID mismatch')
        else:
            wave = load_audio(audio)
            batch = processor.apply_transcription_request(wave, language='English',
                        return_tensors='pt', sampling_rate=16000)
            batch = {k: v.to(device=model.device, dtype=model.dtype if v.is_floating_point() else v.dtype)
                     for k, v in batch.items()}
            with torch.inference_mode():
                generated = model.generate(**batch, max_new_tokens=config['max_new_tokens'], do_sample=False)
            raw = processor.decode(generated[0, batch['input_ids'].shape[-1]:], skip_special_tokens=True)
            transcript = processor.extract_transcription(raw)
            record = {'example_id': key, 'audio_sha256': audio_key, 'raw_text': raw, 'transcription': transcript,
                      'token_ids': generated[0, batch['input_ids'].shape[-1]:].tolist(),
                      'input_tokens': int(batch['input_ids'].shape[-1]),
                      'audio_seconds': len(wave)/16000}
            atomic_json(path, record)
        audio_results[audio_key] = record
        atomic_json(output/'progress.json', {'done': index+1, 'total': len(representatives),
                    'heartbeat_unix': time.time(), 'elapsed_seconds': time.monotonic()-start})
    records = [{**audio_results[r['metadata']['audio_sha256']],
                'example_id': r['example_id'],
                'reference': r['metadata'].get('transcript_reference')} for r in rows]
    distances = []; ref_tokens = []; hyp_tokens = []
    for record in records:
        ref = normalize(record.get('reference') or ''); hyp = normalize(record['transcription'])
        distances.append(edit_distance(ref, hyp)); ref_tokens.extend(ref); hyp_tokens.extend(hyp)
    predictions = output/'predictions.jsonl'
    temporary = predictions.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
    temporary.replace(predictions)
    result = {'status':'passed', 'records':len(records), 'distance':sum(distances),
              'reference_tokens':len(ref_tokens), 'hypothesis_tokens':len(hyp_tokens),
              'wer':sum(distances)/len(ref_tokens) if ref_tokens else None,
              'predictions_sha256':file_hash(predictions),
              'scope':'Qwen3-ASR transcription only; no downstream text decision or model selection'}
    atomic_json(output/'result.json', result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.output/'result.json').exists(): raise FileExistsError('completed run')
        try: run(json.loads(a.config.read_text()),a.output)
        except BaseException as exc:
            atomic_json(a.output/'terminal.json', {'exit_code':1,'error':repr(exc),'finished_unix':time.time()});raise
        atomic_json(a.output/'terminal.json', {'exit_code':0,'finished_unix':time.time()})

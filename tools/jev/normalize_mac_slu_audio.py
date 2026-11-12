#!/usr/bin/env python3
"""Persist explicit 16 kHz float WAV derivatives; keep raw release immutable."""
import argparse
import json
from pathlib import Path
import shutil
import time

from multimodal_jev.baseline import atomic_json, file_hash


def run(source, output):
    import numpy as np
    import scipy
    import soundfile as sf
    from multimodal_jev.sfc_audio import load_sfc_audio
    manifest = json.loads((source/'MANIFEST.json').read_text())
    for name, digest in manifest['files'].items():
        if file_hash(source/name) != digest: raise ValueError('source release changed')
    for name in ('bank.json', 'train.targets.jsonl'): shutil.copyfile(source/name, output/name)
    (output/'audio').mkdir()
    count = 0
    with (output/'train.inputs.jsonl').open('x') as dest:
        for line in (source/'train.inputs.jsonl').open():
            row = json.loads(line); audio = row['audio']
            if audio:
                if file_hash(audio['path']) != audio['sha256']: raise ValueError('raw audio changed')
                wave, receipt = load_sfc_audio(audio['path'])
                path = output/'audio'/Path(audio['path']).name
                sf.write(path, wave, 16000, format='WAV', subtype='FLOAT')
                reloaded, rate = sf.read(path, dtype='float32')
                if rate != 16000 or not np.array_equal(wave, reloaded):
                    raise ValueError('persisted resampling does not reload exactly')
                row['audio'] = dict(path=str(path), sha256=file_hash(path), sample_rate=16000,
                    channels=1, frames=len(wave), duration_seconds=len(wave)/16000,
                    native_span_audio_compatible=True, source=audio, conversion=receipt)
                count += 1
                if count % 1000 == 0:
                    atomic_json(output/'progress.json', dict(completed=count, updated_unix=time.time()))
            dest.write(json.dumps(row, ensure_ascii=False)+'\n')
    result = dict(manifest, schema='mac-slu-span-normalized-v1', source_release=str(source),
        source_manifest_sha256=file_hash(source/'MANIFEST.json'), sample_rates={'16000':count},
        native_audio_compatible=count, files={name:file_hash(output/name) for name in manifest['files']},
        normalization=dict(scipy=scipy.__version__, numpy=np.__version__, soundfile=sf.__version__,
            implementation_sha256=file_hash(Path(__file__).resolve().parents[2]/'src/multimodal_jev/sfc_audio.py'),
            algorithm='scipy.signal.resample_poly, default Kaiser beta 5, ratio 2/3 for observed 24kHz mono source',
            serialization='WAV FLOAT; exact float32 waveform reload checked for every file'),
        normalization_script_sha256=file_hash(Path(__file__)))
    atomic_json(output/'MANIFEST.json', result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args(); a.output.mkdir(exist_ok=False)
    try:
        result=run(a.source,a.output)
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
        print(json.dumps(dict(audio_records=result['native_audio_compatible'],training_admitted=False)))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise

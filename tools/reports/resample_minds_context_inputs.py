"""Versioned 8k-to-16k asset conversion for the unchanged audio consumer."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import resample_poly
import soundfile as sf

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.context_decision_contract import render_context_request
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.slurp_training import content_hash
from tools.reports.prepare_minds_context_inputs import reconstruct


CONVERSION = dict(source_rate=8000, target_rate=16000, algorithm='scipy.signal.resample_poly',
                  up=2, down=1, window=['kaiser', 5.0], padtype='constant', cval=0,
                  output_dtype='float32', wav_subtype='FLOAT', clipping=False, normalization=False)


def convert(waveform, rate):
    if (rate != 8000 or waveform.dtype != np.float32 or waveform.ndim != 1
            or not waveform.size or not np.isfinite(waveform).all()):
        raise ValueError('Finite nonempty original 8k mono float32 required')
    result = resample_poly(waveform, 2, 1, window=('kaiser', 5.0),
                           padtype='constant', cval=0).astype(np.float32)
    if len(result) != 2 * len(waveform) or not np.isfinite(result).all():
        raise ValueError('Complete finite duration-preserving output required')
    return result


def wave_hash(waveform):
    return hashlib.sha256(waveform.astype('<f4').tobytes()).hexdigest()


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():raise ValueError('New immutable output required')
    storage = check_paths({'resampled_inputs': output})
    manifest_hash = file_hash(source / 'MANIFEST.json')
    manifest = json.loads((source / 'MANIFEST.json').read_text())
    if manifest['schema'] != 'minds14-native-descriptor-context-inputs-v1' or manifest['recordings'] != 8168:
        raise ValueError('Complete native descriptor source required')
    for name, digest in manifest['files'].items():
        if file_hash(source / name) != digest:raise ValueError('Source input package changed')
    for path, digest in manifest['source_pins'].items():
        if file_hash(path) != digest:raise ValueError('Source compiler changed')
    definitions = json.loads((source / 'definitions.json').read_text())
    rows = [json.loads(line) for line in (source / 'inputs.jsonl').read_text().splitlines()]
    if len(rows) != 8168 or len({r['id'] for r in rows}) != 8168:raise ValueError('Complete population required')
    for row in rows:reconstruct(row, definitions)
    output.mkdir(parents=True)
    (output / 'audio').mkdir()
    commit_json(output / 'definitions.json', definitions)
    samples = 0
    max_peak = 0.0
    with (output / 'inputs.jsonl').open('x') as inputs, (output / 'conversion.jsonl').open('x') as ledger:
        for index, row in enumerate(rows, 1):
            if file_hash(row['audio']) != row['audio_sha256']:raise ValueError('Original audio changed')
            original, rate = sf.read(row['audio'], dtype='float32')
            converted = convert(original, rate)
            path = output / 'audio' / Path(row['audio']).name
            with path.open('xb') as stream:
                sf.write(stream, converted, 16000, format='WAV', subtype='FLOAT')
                stream.flush();os.fsync(stream.fileno())
            restored, restored_rate = sf.read(path, dtype='float32')
            if restored_rate != 16000 or not np.array_equal(converted, restored):
                raise ValueError('Float WAV serialization changed waveform')
            request = compile_candidates(str(path), **definitions)
            system, text = render_context_request(request)
            target = dict(id=row['id'], audio=str(path), audio_sha256=file_hash(path),
                          request_sha256=content_hash(request.model_input()),
                          rendered_sha256=content_hash(dict(system=system, text=text)))
            if target['rendered_sha256'] != row['rendered_sha256']:raise ValueError('Task text changed')
            inputs.write(json.dumps(target, sort_keys=True) + '\n')
            ledger.write(json.dumps(dict(id=row['id'], source_audio=row['audio'],
                source_audio_sha256=row['audio_sha256'], source_samples=len(original),
                source_waveform_sha256=wave_hash(original), target_audio_sha256=target['audio_sha256'],
                target_samples=len(converted), target_waveform_sha256=wave_hash(converted)), sort_keys=True) + '\n')
            samples += len(converted);max_peak = max(max_peak, float(np.abs(converted).max()))
            if index % 1024 == 0:print(json.dumps(dict(completed=index,total=len(rows))),flush=True)
        for stream in (inputs, ledger):stream.flush();os.fsync(stream.fileno())
    if file_hash(source / 'MANIFEST.json') != manifest_hash:raise ValueError('Source manifest changed')
    fd = os.open(output / 'audio', os.O_RDONLY | os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    result = dict(schema='minds14-native-descriptor-16k-inputs-v1', recordings=len(rows), decisions=14,
                  source_package=dict(path=str(source),manifest_sha256=manifest_hash),
                  protocol=manifest['protocol'], candidate_ids=manifest['candidate_ids'],
                  conversion=CONVERSION, source_pins={str(Path(__file__).resolve()):file_hash(__file__),
                    **manifest['source_pins']}, dependencies=dict(scipy=scipy.__version__,
                    numpy=np.__version__,soundfile=sf.__version__),
                  files={n:file_hash(output / n) for n in ('inputs.jsonl','definitions.json','conversion.jsonl')},
                  rendered_sha256=manifest['rendered_sha256'],hours=samples/16000/3600,max_abs_peak=max_peak,
                  storage=storage,training_dispatch_allowed=False,evaluation_dispatch_allowed=False,
                  scope='Original population, explicit duration-preserving resampling, shared text; no processor or inference.')
    commit_json(output / 'MANIFEST.json', result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=run(a.source,a.output)
    print(json.dumps(dict(recordings=r['recordings'],hours=r['hours'],max_abs_peak=r['max_abs_peak'])))

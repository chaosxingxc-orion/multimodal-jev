"""Re-read the complete original/derived population and recompute the filter.

Independent from the conversion writer; this checks its prescribed scipy
algorithm, not an independent implementation of scipy's resampling filter.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import resample_poly
import soundfile as sf

from multimodal_jev.baseline import file_hash
from tools.reports.prepare_minds_context_inputs import definitions_for, reconstruct


def read(path):
    return json.loads(Path(path).read_text())


def wave_hash(value):
    return hashlib.sha256(value.astype('<f4').tobytes()).hexdigest()


def verify(root, source, protocol_path):
    root, source, protocol_path = map(lambda p: Path(p).resolve(), (root, source, protocol_path))
    m, s, protocol = read(root/'MANIFEST.json'), read(source/'MANIFEST.json'), read(protocol_path)
    expected_conversion = dict(source_rate=8000, target_rate=16000,
        algorithm='scipy.signal.resample_poly', up=2, down=1, window=['kaiser', 5.0],
        padtype='constant', cval=0, output_dtype='float32', wav_subtype='FLOAT',
        clipping=False, normalization=False)
    if (m['schema'] != 'minds14-native-descriptor-16k-inputs-v1'
            or s['schema'] != 'minds14-native-descriptor-context-inputs-v1'
            or m['source_package'] != dict(path=str(source), manifest_sha256=file_hash(source/'MANIFEST.json'))
            or m['protocol'] != s['protocol']
            or m['protocol'] != dict(path=str(protocol_path), sha256=file_hash(protocol_path))
            or m['conversion'] != expected_conversion or m['recordings'] != 8168
            or s['recordings'] != 8168 or m['decisions'] != 14
            or m['training_dispatch_allowed'] or m['evaluation_dispatch_allowed']):
        raise ValueError('Complete frozen preparation-only protocol required')
    for folder, manifest in ((root, m), (source, s)):
        for name, digest in manifest['files'].items():
            if file_hash(folder/name) != digest:raise ValueError('Package file changed')
        for name, digest in manifest['source_pins'].items():
            if file_hash(name) != digest:raise ValueError('Producer dependency changed')
    if m['dependencies'] != dict(scipy=scipy.__version__, numpy=np.__version__, soundfile=sf.__version__):
        raise ValueError('Recomputation library versions differ')
    definitions = definitions_for(protocol)
    if read(root/'definitions.json') != definitions or read(source/'definitions.json') != definitions:
        raise ValueError('Candidate definitions changed')
    rows = [json.loads(line) for line in (root/'inputs.jsonl').read_text().splitlines()]
    originals = [json.loads(line) for line in (source/'inputs.jsonl').read_text().splitlines()]
    ledger = [json.loads(line) for line in (root/'conversion.jsonl').read_text().splitlines()]
    if len(rows) != 8168 or len({r['id'] for r in rows}) != 8168:
        raise ValueError('Complete unique population required')
    samples = 0
    max_peak = 0.0
    for index, (r, original, entry) in enumerate(zip(rows, originals, ledger, strict=True), 1):
        reconstruct(r, definitions); reconstruct(original, definitions)
        if r['id'] != original['id'] or r['rendered_sha256'] != original['rendered_sha256']:
            raise ValueError('Population order or task text changed')
        target = root/'audio'/(r['id'].removeprefix('minds14:')+'.wav')
        if (target.is_symlink() or not target.resolve().is_relative_to(root/'audio')
                or r['audio'] != str(target) or file_hash(target) != r['audio_sha256']
                or file_hash(original['audio']) != original['audio_sha256']):
            raise ValueError('Audio path or encoded bytes changed')
        x, xr = sf.read(original['audio'], dtype='float32')
        y, yr = sf.read(target, dtype='float32')
        if (xr != 8000 or yr != 16000 or x.ndim != 1 or y.ndim != 1
                or len(y) != 2*len(x) or sf.info(target).subtype != 'FLOAT'
                or not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError('Duration, format or finite waveform mismatch')
        expected = resample_poly(x, 2, 1, window=('kaiser',5.0), padtype='constant', cval=0).astype(np.float32)
        if not np.array_equal(y, expected):raise ValueError('Filter recomputation differs')
        if entry != dict(id=r['id'], source_audio=original['audio'],
                source_audio_sha256=original['audio_sha256'], source_samples=len(x),
                source_waveform_sha256=wave_hash(x), target_audio_sha256=r['audio_sha256'],
                target_samples=len(y), target_waveform_sha256=wave_hash(y)):
            raise ValueError('Conversion ledger differs from reread waveforms')
        samples += len(y); max_peak = max(max_peak, float(np.abs(y).max()))
        if index % 1024 == 0:print(json.dumps(dict(verified=index,total=8168)),flush=True)
    names = {r['id'].removeprefix('minds14:')+'.wav' for r in rows}
    if {p.name for p in (root/'audio').iterdir()} != names:
        raise ValueError('Unexpected derived audio files')
    if m['hours'] != samples/16000/3600 or m['max_abs_peak'] != max_peak:
        raise ValueError('Aggregate statistics differ')
    return dict(schema='minds14-resampling-independent-verification-v1', root=str(root),
        manifest_sha256=file_hash(root/'MANIFEST.json'), source_manifest_sha256=file_hash(source/'MANIFEST.json'),
        protocol_sha256=file_hash(protocol_path), verifier_sha256=file_hash(__file__),
        recordings=len(rows), hours=samples/16000/3600, max_abs_peak=max_peak,
        complete_population_and_order=True, complete_waveforms_recomputed_exact=True,
        full_duration_preserved=True, task_text_unchanged=True, no_target_annotations_opened=True,
        training_dispatch_allowed=False, evaluation_dispatch_allowed=False,
        scope='Full original/derived reread and same-library filter recomputation; no processor or inference.')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('root','source','protocol','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();result=verify(a.root,a.source,a.protocol)
    from multimodal_jev.local_checkpoint import commit_json
    if Path(a.output).exists():raise ValueError('New verification receipt required')
    commit_json(Path(a.output),result)
    print(json.dumps(result,indent=2))

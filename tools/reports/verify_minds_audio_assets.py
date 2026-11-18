"""Independently re-read all extracted MINDS files and reconcile audit population."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    if (manifest['schema'] != 'minds14-label-free-audio-assets-v1'
            or manifest['training_dispatch_allowed'] or manifest['evaluation_dispatch_allowed']):
        raise ValueError('Audio preparation only')
    audit_path = Path(manifest['audit']['path'])
    if digest(audit_path) != manifest['audit']['sha256']:
        raise ValueError('Audit identity changed')
    audit = json.loads(audit_path.read_text())
    expected = {r['id']: r for r in audit['records']}
    if len(expected) != len(audit['records']) or len(expected) != 8168:
        raise ValueError('Complete unique audited population required')
    if digest(root / 'inputs.jsonl') != manifest['inputs_sha256']:
        raise ValueError('Input inventory changed')
    seen = set()
    samples = 0
    for line in (root / 'inputs.jsonl').read_text().splitlines():
        r = json.loads(line)
        if set(r) != {'id', 'audio', 'audio_sha256', 'sample_rate', 'channels', 'samples', 'waveform_sha256'}:
            raise ValueError('Unexpected consumer fields')
        if r['id'] in seen or r['id'] not in expected:
            raise ValueError('Unexpected/duplicate recording')
        e = expected[r['id']]
        safe_name = 'audio/' + r['id'].removeprefix('minds14:') + '.wav'
        path = root / r['audio']
        if r['audio'] != safe_name or path.is_symlink() or not path.resolve().is_relative_to(root / 'audio'):
            raise ValueError('Unsafe or label-bearing audio path')
        if digest(path) != r['audio_sha256'] or r['audio_sha256'] != e['encoded_audio_sha256']:
            raise ValueError('Encoded bytes differ from audited original')
        waveform, rate = sf.read(path, dtype='float32', always_2d=True)
        identity = f'{rate}:{waveform.shape[1]}:'.encode() + waveform.astype('<f4').tobytes()
        if (rate != 8000 or r['sample_rate'] != rate or e['sample_rate'] != rate
                or waveform.shape[1] != 1 or r['channels'] != 1 or e['channels'] != 1
                or len(waveform) != r['samples'] or len(waveform) != e['samples']
                or not np.isfinite(waveform).all()
                or hashlib.sha256(identity).hexdigest() != e['waveform_sha256']
                or r['waveform_sha256'] != e['waveform_sha256']):
            raise ValueError('Decoded waveform mismatch')
        samples += len(waveform)
        seen.add(r['id'])
    if seen != set(expected) or manifest['recordings'] != len(seen):
        raise ValueError('Incomplete population')
    names = {'audio/' + uid.removeprefix('minds14:') + '.wav' for uid in seen}
    if {str(p.relative_to(root)) for p in (root / 'audio').iterdir()} != names:
        raise ValueError('Unexpected audio directory contents')
    return dict(schema='minds14-audio-assets-independent-verification-v1', root=str(root),
                manifest_sha256=digest(root / 'MANIFEST.json'), inputs_sha256=digest(root / 'inputs.jsonl'),
                verifier_sha256=digest(Path(__file__)), recordings=len(seen), hours=samples/8000/3600,
                full_encoded_bytes_equal_audited_original=True, full_decoded_waveforms_equal_audit=True,
                exact_consumer_schema_no_labels_or_transcripts=True,
                training_dispatch_allowed=False, evaluation_dispatch_allowed=False,
                scope='Complete local audio extraction only; no resampling, prompt, processor or model validation.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = verify(a.root)
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))

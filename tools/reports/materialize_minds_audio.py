"""Extract audited MINDS audio without label-bearing paths in consumer inputs.

No task templates, resampling, transcript reads, model inference or dispatch.
The final manifest is published only after every source/record is checked.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from multimodal_jev.local_runtime_storage import check_paths


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_record(row, language, expected):
    original = row['path']
    uid = 'minds14:' + hashlib.sha256((language + '\0' + original).encode()).hexdigest()
    payload = row['audio']['bytes']
    if (uid != expected['id'] or language != expected['language']
            or hashlib.sha256(original.encode()).hexdigest() != expected['source_path_sha256']
            or hashlib.sha256(payload).hexdigest() != expected['encoded_audio_sha256']):
        raise ValueError('Source/audio identity differs from audited record')
    with sf.SoundFile(io.BytesIO(payload)) as stream:
        if stream.format != 'WAV':
            raise ValueError('Expected original WAV payload')
        rate = stream.samplerate
        waveform = stream.read(dtype='float32', always_2d=True)
    identity = f'{rate}:{waveform.shape[1]}:'.encode() + waveform.astype('<f4').tobytes()
    if (not len(waveform) or not np.isfinite(waveform).all()
            or rate != expected['sample_rate'] or rate != 8000
            or waveform.shape[1] != expected['channels'] or waveform.shape[1] != 1
            or len(waveform) != expected['samples']
            or hashlib.sha256(identity).hexdigest() != expected['waveform_sha256']):
        raise ValueError('Decoded waveform differs from audited record')
    # Fixed schema: never forward source path, intent, transcript or audio metadata.
    return payload, dict(id=uid, audio='audio/' + uid.split(':')[1] + '.wav',
                         audio_sha256=expected['encoded_audio_sha256'],
                         sample_rate=rate, channels=1, samples=len(waveform),
                         waveform_sha256=expected['waveform_sha256'])


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_file(path, payload):
    with path.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def materialize(root, audit_path, output):
    import pyarrow
    import pyarrow.parquet as pq

    root, audit_path, output = map(Path, (root, audit_path, output))
    if output.exists():
        raise ValueError('New output required; incomplete outputs are never treated as published')
    storage = check_paths({'audio_output': output})
    audit_hash = sha_file(audit_path)
    audit = json.loads(audit_path.read_text())
    expected = {r['id']: r for r in audit['records']}
    if (audit['schema'] != 'minds-local-audio-audit-v1'
            or len(expected) != len(audit['records']) or len(expected) != 8168
            or any('decode_error' in r for r in expected.values())):
        raise ValueError('Complete successfully decoded unique audit population required')
    files = sorted(root.glob('*/train-*.parquet'))
    audited_files = {str(Path(p).resolve()) for p in audit['sources'] if p.endswith('.parquet')}
    if len(files) != 14 or {str(p.resolve()) for p in files} != audited_files:
        raise ValueError('Exact audited source shard set required')
    for path in files:
        if sha_file(path) != audit['sources'].get(str(path)):
            raise ValueError('Source shard changed')
    output.mkdir(parents=True)
    (output / 'audio').mkdir()
    seen = set()
    inputs = []
    for path in files:
        # Read only path and bytes. Gold remains in the separately bound audit.
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=['path', 'audio']):
            for row in batch.to_pylist():
                uid = 'minds14:' + hashlib.sha256((path.parent.name + '\0' + row['path']).encode()).hexdigest()
                if uid in seen or uid not in expected:
                    raise ValueError('Duplicate or unaudited record')
                payload, item = validate_record(row, path.parent.name, expected[uid])
                write_file(output / item['audio'], payload)
                inputs.append(item)
                seen.add(uid)
        print(json.dumps(dict(shard=path.parent.name, completed=len(seen))), flush=True)
    if seen != set(expected):
        raise ValueError('Incomplete population')
    # Catch source replacement while reading before publishing success.
    for path in files:
        if sha_file(path) != audit['sources'][str(path)]:
            raise ValueError('Source shard changed during extraction')
    if sha_file(audit_path) != audit_hash:
        raise ValueError('Audit changed during extraction')
    inputs.sort(key=lambda r: r['id'])
    write_file(output / 'inputs.jsonl', ''.join(json.dumps(r, sort_keys=True) + '\n' for r in inputs).encode())
    sync_dir(output / 'audio')
    manifest = dict(schema='minds14-label-free-audio-assets-v1', recordings=len(inputs),
                    audit=dict(path=str(audit_path.resolve()), sha256=audit_hash),
                    source_shards={str(p): audit['sources'][str(p)] for p in files},
                    inputs_sha256=sha_file(output / 'inputs.jsonl'),
                    tool_sha256=sha_file(Path(__file__)), storage=storage,
                    dependencies=dict(pyarrow=pyarrow.__version__, numpy=np.__version__, soundfile=sf.__version__),
                    training_dispatch_allowed=False, evaluation_dispatch_allowed=False,
                    scope='Original 8kHz mono WAV bytes, opaque paths, complete upstream TRAIN; '
                          'no resampling, transcript access, prompt, processor or inference. '
                          'Labels/language metadata stay in the separately bound audit, not consumer inputs.')
    write_file(output / 'MANIFEST.json', (json.dumps(manifest, indent=2) + '\n').encode())
    sync_dir(output)
    sync_dir(output.parent)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = materialize(args.root, args.audit, args.output)
    print(json.dumps(dict(recordings=result['recordings'], inputs_sha256=result['inputs_sha256'])))

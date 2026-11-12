#!/usr/bin/env python3
"""Prepare immutable TRAIN audio/targets and a fixed finite Span menu.

No missing negative labels are invented. This is a diagnostic data release,
not permission to optimize on partial or unreviewed annotation.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.mac_slu_projection import convert_annotation, declared_bank, official_frames


def run(source, official_source, expected_audit, output):
    import numpy as np
    import soundfile as sf
    started = time.time()
    audit = json.loads(expected_audit.read_text())
    for path, digest in audit['sources'].items():
        if file_hash(path) != digest: raise ValueError('audited source changed')
    if file_hash(official_source) != audit['official_prompt_source_sha256']:
        raise ValueError('official definitions changed')
    labels = source / 'label/train_set.jsonl'
    archive_path = source / 'audio_train.tar.gz'
    if str(labels) not in audit['sources'] or str(archive_path) not in audit['sources']:
        raise ValueError('source must match audit')
    rows = [json.loads(line) for line in labels.open()]
    converted = [convert_annotation(r) for r in rows]
    if len({r['id'] for r in rows}) != len(rows): raise ValueError('duplicate source ids')
    bank = declared_bank(official_source.read_text(), converted)
    atomic_json(output / 'bank.json', bank)
    (output / 'audio').mkdir()
    audio, rates, channels = {}, Counter(), Counter()
    with tarfile.open(archive_path, 'r|gz') as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts: raise ValueError('unsafe archive path')
            if member.isdir(): continue
            if (not member.isfile() or len(path.parts) != 2 or path.parts[0] != 'audio_train'
                    or path.suffix != '.wav' or not path.stem.startswith('id_')):
                raise ValueError('unexpected archive member')
            uid = path.stem[3:]
            if uid in audio: raise ValueError('duplicate audio id')
            data = archive.extractfile(member).read()
            signal, rate = sf.read(io.BytesIO(data), dtype='float32', always_2d=True)
            if not signal.size or not np.isfinite(signal).all(): raise ValueError('invalid waveform')
            dest = output / 'audio' / path.name
            with dest.open('xb') as handle: handle.write(data)
            audio[uid] = dict(path=str(dest), sha256=hashlib.sha256(data).hexdigest(),
                sample_rate=rate, channels=signal.shape[1], frames=signal.shape[0],
                duration_seconds=signal.shape[0] / rate,
                native_span_audio_compatible=rate == 16000 and signal.shape[1] == 1)
            rates[rate] += 1; channels[signal.shape[1]] += 1
            if len(audio) % 1000 == 0:
                atomic_json(output / 'progress.json', dict(stage='decoding_audio', records=len(audio), updated_unix=time.time()))
    missing = sorted({r['id'] for r in rows} - set(audio))
    if missing != audit['audio']['missing_audio_ids'] or set(audio)-{r['id'] for r in rows}:
        raise ValueError('audio population differs from audit')
    counts = Counter()
    with (output / 'train.inputs.jsonl').open('x') as inputs, (output / 'train.targets.jsonl').open('x') as targets:
        for raw, target in zip(rows, converted, strict=True):
            uid = raw['id']
            entry = audio.get(uid)
            request = dict(id=uid, split='train', audio=entry, bank_id=bank['bank_id'],
                           expansion='all_fixed_bank_intents_each_with_all_domain_roles',
                           training_admitted=False)
            # Inputs carry no source text, target-derived subset, or missing-label flag.
            inputs.write(json.dumps(request, ensure_ascii=False) + '\n')
            try:
                official = dict(status='representable', semantics=official_frames(target))
            except ValueError as exc:
                official = dict(status='unavailable', reason=str(exc), semantics=None)
            counts[official['status']] += 1
            targets.write(json.dumps(dict(**target, official_format=official), ensure_ascii=False) + '\n')
    files = {p.name: file_hash(p) for p in (output / 'bank.json', output / 'train.inputs.jsonl', output / 'train.targets.jsonl')}
    manifest = dict(schema='mac-slu-span-preparation-v1', split='train', records=len(rows),
        audio_records=len(audio), missing_audio_ids=missing, sample_rates=dict(rates), channels=dict(channels),
        audio_seconds=sum(v['duration_seconds'] for v in audio.values()),
        native_audio_compatible=sum(v['native_span_audio_compatible'] for v in audio.values()),
        official_format_counts=dict(counts), domain_intents=len(bank['intents']), domain_roles=len(bank['roles']),
        files=files, source_files=audit['sources'], official_source_sha256=file_hash(official_source),
        audit_sha256=file_hash(expected_audit), training_admitted=False,
        negative_supervision='unavailable_pending_annotation_completeness_review',
        task_scope='Parallel intent and bound-role presence, one forward per fixed intent group; '
                   'not decision-chain reasoning, instance-value extraction or official full SLU.',
        elapsed_seconds=time.time()-started,
        code={str(p): file_hash(p) for p in (Path(__file__), Path(__file__).resolve().parents[2]/'src/multimodal_jev/mac_slu_projection.py')})
    atomic_json(output / 'MANIFEST.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('source', 'official-source', 'audit', 'output'): parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    try:
        result = run(args.source, args.official_source, args.audit, args.output)
        atomic_json(args.output/'terminal.json', dict(exit_code=0, finished_unix=time.time()))
        print(json.dumps(result, ensure_ascii=False))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json', dict(exit_code=1, error=repr(exc), finished_unix=time.time()))
        raise

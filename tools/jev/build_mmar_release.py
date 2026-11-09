#!/usr/bin/env python3
"""Safely extract and freeze the complete MMAR benchmark with original options."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import numpy as np
import soundfile as sf
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.mmar import model_input


def build(root, destination):
    source = root / 'MMAR-meta.json'; archive = root / 'mmar-audio.tar.gz'
    rows = json.loads(source.read_text())
    if len(rows) != 1000 or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('expected 1000 unique publisher questions')
    paths = set()
    for row in rows:
        model_input(row)
        path = PurePosixPath(row['audio_path'])
        if path.is_absolute() or '..' in path.parts or path.parts[0] != 'audio':
            raise ValueError('unsafe source audio path')
        paths.add(str(path))
    # Keep audio once under the original shared dataset, not in each release.
    extracted = root / 'extracted'
    if not extracted.exists():
        stage = Path(tempfile.mkdtemp(prefix='.extract-', dir=root))
        try:
            seen = set()
            with tarfile.open(archive, 'r:gz') as tar:
                for member in tar:
                    path = PurePosixPath(member.name)
                    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != 'audio':
                        raise ValueError('unsafe archive path')
                    if member.isdir():
                        continue
                    if not member.isfile() or str(path) in seen:
                        raise ValueError('non-regular or duplicate archive member')
                    seen.add(str(path))
                    target = stage / str(path); target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as inp, target.open('xb') as out:
                        shutil.copyfileobj(inp, out)
            if seen != paths:
                raise ValueError('archive/source inventory mismatch')
            os.rename(stage, extracted)
        except BaseException:
            shutil.rmtree(stage)
            raise
    inventory = {}
    for relative in sorted(paths):
        audio = extracted / relative
        with sf.SoundFile(audio) as reader:
            frames, rate, channels = len(reader), reader.samplerate, reader.channels
            if frames <= 0 or rate <= 0:
                raise ValueError('invalid audio')
            decoded = 0
            for block in reader.blocks(blocksize=65536, dtype='float32', always_2d=True):
                if not np.isfinite(block).all():
                    raise ValueError('nonfinite audio samples')
                decoded += len(block)
            if decoded != frames:
                raise ValueError('truncated audio')
        inventory[relative] = {'sha256': file_hash(audio), 'frames': frames,
                               'sample_rate': rate, 'channels': channels}
    destination.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.mmar-', dir=destination))
    try:
        with (stage / 'inputs.jsonl').open('w') as out:
            for row in rows:
                inp = model_input(row); relative = str(PurePosixPath(inp['audio']))
                inp['audio'] = str(extracted / relative)
                record = {'id': row['id'], 'input': inp, 'audio': inventory[relative]}
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
        shutil.copyfile(source, stage / 'references.json')
        atomic_json(stage / 'audio-inventory.json', inventory)
        receipt = {'protocol': 'mmar-native-v1', 'questions': len(rows), 'audio_files': len(inventory),
                   'source_sha256': file_hash(source), 'archive_sha256': file_hash(archive),
                   'publisher_scorer_sha256': file_hash(root / 'code/evaluation.py'),
                   'builder_sha256': file_hash(__file__),
                   'input_code_sha256': file_hash(Path(__file__).parents[2] / 'src/multimodal_jev/mmar.py'),
                   'inputs_sha256': file_hash(stage / 'inputs.jsonl'),
                   'inventory_sha256': file_hash(stage / 'audio-inventory.json'),
                   'empty_candidate_questions': sum(any(not s.strip() for s in r['choices']) for r in rows),
                   'reference_not_single_option': sum(r['answer'] not in r['choices'] for r in rows),
                   'scope': 'all 1000 official test questions; original candidates including padding; no model inference'}
        release = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
        receipt['release_id'] = release; final = destination / release
        if final.exists():
            raise FileExistsError(final)
        atomic_json(stage / 'MANIFEST.json', receipt); os.rename(stage, final)
        return final, receipt
    except BaseException:
        shutil.rmtree(stage)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    path, receipt = build(args.root, args.output_root)
    print(json.dumps({'path': str(path), **receipt}, indent=2))

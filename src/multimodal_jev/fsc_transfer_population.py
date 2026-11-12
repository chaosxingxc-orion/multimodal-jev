"""Official FSC audio population, with labels exposed only to offline scoring."""
import csv
import io
from pathlib import Path, PurePosixPath

from .baseline import file_hash

SPLITS = ('train', 'valid', 'test')


def official_sources(root, expected_hashes):
    root = Path(root).resolve()
    if set(expected_hashes) != set(SPLITS):
        raise ValueError('all official split hashes required')
    tables = {}; paths = {}; speakers = {}
    for split in SPLITS:
        path = root/'data'/f'{split}_data.csv'
        if file_hash(path) != expected_hashes[split]:
            raise ValueError('official CSV identity differs')
        rows = list(csv.DictReader(io.StringIO(path.read_text())))
        if not rows:
            raise ValueError('empty official split')
        for row in rows:
            rel = PurePosixPath(row['path']); speaker = row['speakerId']
            if (rel.is_absolute() or '..' in rel.parts or str(rel) != row['path']
                    or len(rel.parts) != 4 or rel.parts[:2] != ('wavs', 'speakers')
                    or rel.parts[2] != speaker or not speaker or rel.suffix != '.wav'
                    or not (root/str(rel)).resolve().is_relative_to(root)):
                raise ValueError('canonical in-dataset speaker audio path required')
            if str(rel) in paths:
                raise ValueError('duplicate or cross-split recording')
            if speaker in speakers and speakers[speaker] != split:
                raise ValueError('cross-split speaker overlap')
            paths[str(rel)] = split; speakers[speaker] = split
        tables[split] = rows
    return tables


def build_population(root, split, expected_hashes):
    if split not in ('valid', 'test'):
        raise ValueError('full official evaluation split required')
    root = Path(root).resolve(); tables = official_sources(root, expected_hashes)
    rows = []; seen_hashes = set()
    for row in tables[split]:
        audio = root/row['path']; digest = file_hash(audio)
        if digest in seen_hashes:
            raise ValueError('duplicate audio within evaluation population')
        seen_hashes.add(digest)
        rows.append(dict(id='fsc:'+row['path'], audio=str(audio), audio_sha256=digest,
                         speaker_id=row['speakerId']))
    return dict(schema='fsc-official-span-population-v1', role='frozen_transfer_evaluation_inputs',
        split=split, dataset_root=str(root), source_csv_sha256=dict(expected_hashes),
        records=rows, recordings=len(rows), speakers=len({r['speaker_id'] for r in rows}),
        selection='complete official split in published CSV order; no label-based selection',
        scope='Label-free input roster only. No model inference, checkpoint selection or transfer efficacy. '
              'Cross-split path/speaker disjointness checked; cross-split waveform duplication not certified here.')


def scoring_targets(population, task):
    """Use only after source checkpoints and evaluation protocol are frozen."""
    from .fsc_span_transfer import validate, FIELDS
    validate(task)
    if (population['schema'] != 'fsc-official-span-population-v1'
            or population['role'] != 'frozen_transfer_evaluation_inputs'
            or population['split'] not in ('valid', 'test')):
        raise ValueError('official evaluation population required')
    root = Path(population['dataset_root']).resolve()
    source = official_sources(root, population['source_csv_sha256'])[population['split']]
    roster = population['records']
    if (population['recordings'] != len(source) or len(roster) != len(source)
            or population['speakers'] != len({r['speakerId'] for r in source})):
        raise ValueError('complete official population required')
    commands = [c['values'] for c in task['commands']]; targets = []
    for row, item in zip(source, roster, strict=True):
        audio = root/row['path']
        if (item['id'] != 'fsc:'+row['path'] or item['audio'] != str(audio)
                or item['speaker_id'] != row['speakerId'] or item['audio_sha256'] != file_hash(audio)):
            raise ValueError('published roster, speaker or audio identity changed')
        gold = {f: row[f] for f in FIELDS}
        if gold not in commands:
            raise ValueError('official target outside TRAIN-defined task ontology')
        targets.append(dict(id=item['id'], gold=gold))
    return targets

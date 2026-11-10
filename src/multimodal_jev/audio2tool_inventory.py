"""Answer-free Audio2Tool asset indexing and independently pinned tool registry.

Does not load query metadata, annotations, transcripts or audio waveforms.
Inventory order is an explicit local protocol, not a reconstructed paper recipe.
"""
import csv
import hashlib
import io
from pathlib import PurePosixPath
import re

REVISION = 'f1388da9a3189541ab82adac88824a0661670c43'
REGISTRY_FIELDS = ('tool_id', 'domain', 'category', 'tool_name', 'signature',
                   'description', 'argument_defaults', 'argument_constraints')


def registry(data, expected_sha256):
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('registry hash differs')
    reader = csv.DictReader(io.StringIO(data.decode('utf-8')))
    if tuple(reader.fieldnames or ()) != REGISTRY_FIELDS:
        raise ValueError('unexpected registry fields')
    rows = list(reader)
    if not rows or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError('incomplete registry rows')
    for key in ('tool_id', 'tool_name'):
        if any(not r[key] for r in rows) or len({r[key] for r in rows}) != len(rows):
            raise ValueError('empty or duplicate tool identity')
    if any(not r['tool_id'].isdigit() or not r['signature'].startswith(r['tool_name']+'(')
           or not r['description'] for r in rows):
        raise ValueError('invalid tool definition')
    # Preserve CSV order and all definition fields, including defaults/constraints.
    return rows


def audio_inventory(files):
    groups = {}; paths = set()
    prefix = 'datasets/audio2tool/'
    for row in files:
        if row.get('asset') != 'audio2tool':
            continue
        name = row['path']; path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or not name.startswith(prefix):
            raise ValueError('invalid release path')
        if name in paths:
            raise ValueError('duplicate asset path')
        paths.add(name)
        relative = name[len(prefix):]
        if row['url'] != f'https://huggingface.co/datasets/RVtech/Audio2Tool/resolve/{REVISION}/{relative}':
            raise ValueError('unpinned asset URL')
        if path.suffix != '.wav':
            continue
        parts = path.parts
        if len(parts) != 6 or parts[2] != 'public' or not re.fullmatch(r'query_\d+', parts[4]):
            raise ValueError('unknown audio layout')
        match = re.fullmatch(r'tier([1-8])_[a-z_]+_audios', parts[3])
        if not match or row['algorithm'] != 'sha256' or not re.fullmatch('[0-9a-f]{64}',row['hash']):
            raise ValueError('tier and SHA256 audio pin required')
        if type(row['size']) is not int or row['size'] <= 0:
            raise ValueError('positive audio byte length required')
        tier = int(match[1]); key = (tier, parts[4])
        audio = dict(path=relative, sha256=row['hash'], bytes=row['size'])
        if tier == 7:
            turn = re.fullmatch(r'turn_(\d+)_(user|agent)_(.+)\.wav', parts[5])
            if not turn:
                raise ValueError('unknown conversation turn filename')
            audio.update(turn=int(turn[1]),role=turn[2])
        elif tier == 8:
            if parts[5] != 'mixed.wav':raise ValueError('unknown mixed audio filename')
        elif not re.fullmatch(r'speaker_\d+_.+\.wav',parts[5]):
            raise ValueError('unknown speaker variant filename')
        groups.setdefault(key, []).append(audio)
    if not groups:raise ValueError('empty audio inventory')
    result=[]
    for (tier, query), audios in sorted(groups.items()):
        if tier == 7:
            if len({a['turn'] for a in audios}) != len(audios):
                raise ValueError('ambiguous duplicate conversation turn')
            audios.sort(key=lambda a:a['turn'])
        else:audios.sort(key=lambda a:a['path'])
        if tier == 8 and len(audios)!=1:raise ValueError('one mixed waveform per query required')
        result.append(dict(tier=tier,query_id=query,
            audio_semantics='ordered_conversation_turns' if tier==7 else 'independent_variants',
            audios=audios))
    return result

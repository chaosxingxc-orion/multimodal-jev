"""TRAIN-only source closure; no rule generation, scoring or model inputs."""
import hashlib
import json
import unicodedata
from collections import defaultdict

from .slurp_native import native_target

PROTOCOL = 'slurp-clean-v7-source80-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def partition(metadata, requests):
    """Keep utterance, normalized text, filename and saved audio-hash closure.

    Callers authenticate metadata/request files. Input requests must contain the
    entire original TRAIN roster, including clean-v7 exclusions; only retained
    metadata is exported. No waveform equivalence or speaker isolation claim.
    """
    by_id = {}
    for row in metadata:
        uid = str(row['slurp_id'])
        if uid in by_id:
            raise ValueError('duplicate source ID')
        by_id[uid] = row
    if not by_id:
        raise ValueError('nonempty TRAIN metadata required')
    index = {}
    for req in requests:
        if req['split'] != 'train' or req['id'] in index:
            raise ValueError('unique TRAIN requests required')
        index[req['id']] = req
    parent = {uid: uid for uid in by_id}

    def find(uid):
        while parent[uid] != uid:
            parent[uid] = parent[parent[uid]]
            uid = parent[uid]
        return uid

    def join(a, b):
        a, b = sorted((find(a), find(b)))
        parent[b] = a

    owners = {}; files = defaultdict(list); hash_files = defaultdict(set)
    for uid, row in sorted(by_id.items()):
        text = ' '.join(unicodedata.normalize('NFKC', row['sentence']).casefold().split())
        if not text:
            raise ValueError('empty source text')
        keys = [('text', digest(text))]
        target = native_target(row)
        for rec in row['recordings']:
            filename = rec['file']; req = index.get('slurp:' + filename)
            if req is None:
                raise ValueError('retained recording missing from native roster')
            audio = req['audio_sha256']
            if not isinstance(audio, str) or len(audio) != 64 or any(c not in '0123456789abcdef' for c in audio):
                raise ValueError('invalid audio hash')
            files[filename].append((uid, target))
            hash_files[audio].add(filename)
            keys.extend([('file', filename), ('audio', audio)])
        for key in keys:
            if key in owners:
                join(uid, owners[key])
            else:
                owners[key] = uid
    components = defaultdict(list)
    for uid in sorted(by_id):
        components[find(uid)].append(uid)
    group_for = {}
    groups = []
    for members in sorted(components.values()):
        group = 'slurp-clean:' + digest(members)
        bucket = int.from_bytes(hashlib.sha256((PROTOCOL + ':' + group).encode()).digest()[:8], 'big') % 10000
        role = 'learning' if bucket < 8000 else 'selection'
        groups.append(dict(source_group=group, source_ids=members, role=role))
        for uid in members:
            group_for[uid] = (group, role)
    quarantine = set()
    for filenames in hash_files.values():
        targets = {digest(t) for f in filenames for _, t in files[f]}
        if len(targets) > 1:
            quarantine.update(filenames)
    records = []
    for filename, occurrences in sorted(files.items()):
        uid = occurrences[0][0]; group, role = group_for[uid]
        records.append(dict(id='slurp:' + filename, source_group=group,
            source_ids=sorted({u for u, _ in occurrences}), role=role,
            audio_sha256=index['slurp:' + filename]['audio_sha256'],
            supervision_status='conflict' if filename in quarantine else 'available'))
    return dict(protocol=PROTOCOL, groups=groups, records=records,
        quarantine_ids=sorted('slurp:' + f for f in quarantine),
        speaker_disjoint=False, semantic_near_duplicate_isolation=False,
        optimization_ready=False)

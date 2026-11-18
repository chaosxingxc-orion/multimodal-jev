"""Resolve the immutable publication dependencies already used by Stage1."""
import hashlib
import json
from pathlib import Path

RELEASE = Path('/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/official-clean-intent-v1')
RELEASE_SHA = '38e4273d9328e4f02db081315e46f97a17628b974be7ba175254882e95421482'
DONOR = Path('.local/simple-final-v1/prefix-prepared.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publication_mounts(release=RELEASE, digest=RELEASE_SHA, donor=DONOR):
    release = Path(release)
    if sha(release/'MANIFEST.json') != digest:
        raise ValueError('publication identity changed')
    manifest = json.loads((release/'MANIFEST.json').read_text())
    if sha(release/'publication-config.json') != manifest['files']['publication-config.json']:
        raise ValueError('publication dependency identity changed')
    publication = json.loads((release/'publication-config.json').read_text())
    argv = json.loads(Path(donor).read_text())['argv']
    mounts = {}
    for i, arg in enumerate(argv):
        if arg == '-v':
            parts = argv[i+1].split(':')
            if len(parts) not in (2, 3):
                raise ValueError('invalid donor bind')
            source, target = parts[:2]
            mode = parts[2] if len(parts) == 3 else 'rw'
            if target in publication['pins']:
                if mode != 'ro' or target in mounts:
                    raise ValueError('unique read-only dependency mount required')
                mounts[target] = (source, mode)
    result = []
    for target, expected in publication['pins'].items():
        if sha(target) != expected:
            raise ValueError('local publication dependency changed: '+target)
        if target.startswith('/mnt/cephfs/xingchao/'):
            continue  # Existing read-only dataset mount supplies these paths.
        if target not in mounts:
            raise ValueError('publication dependency mount missing: '+target)
        source, mode = mounts[target]
        if not source.startswith('/mnt/cephfs/xingchao/') or sha(source) != expected:
            raise ValueError('published mount source differs: '+source)
        result.append(source+':'+target+':'+mode)
    return result

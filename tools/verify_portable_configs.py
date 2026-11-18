"""Read-only exact JSON configuration comparison against pinned source files.

Uses the shared macro resolver. Does not validate referenced assets, dispatch
jobs, score predictions, or permit differences in source configuration values.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

from multimodal_jev.local_config import load_config


def encoded(value):
    # Unlike Python equality this distinguishes false/0 and 1/1.0. Key order
    # and whitespace are immaterial; list order and numeric types are retained.
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def verify(config_path, local):
    config, provenance = load_config(config_path, local)
    if (set(config) != {'schema', 'entries'} or config['schema'] != 'portable-config-verification-v1'
            or not isinstance(config['entries'], list) or not config['entries']):
        raise ValueError('Nonempty portable configuration verification entries required')
    sources = {str(Path(config_path).resolve()): provenance['template_sha256'],
               str(Path(local).resolve()): provenance['local_bindings_sha256']}
    reports, ids = [], set()

    def bind(path, digest):
        path = str(Path(path).resolve())
        if path in sources and sources[path] != digest:
            raise ValueError('Conflicting source identity')
        sources[path] = digest

    for name in (__file__, 'src/multimodal_jev/local_config.py'):
        bind(name, hashlib.sha256(Path(name).read_bytes()).hexdigest())
    for row in config['entries']:
        if (set(row) != {'id','template','source','source_sha256'}
                or not isinstance(row['id'], str) or not re.fullmatch(r'[a-z][a-z0-9_-]*',row['id'])
                or row['id'] in ids or not isinstance(row['source_sha256'], str)
                or not re.fullmatch(r'[0-9a-f]{64}',row['source_sha256'])):
            raise ValueError('Distinct IDs and pinned source identities required')
        ids.add(row['id'])
        source = Path(row['source']).read_bytes()
        if hashlib.sha256(source).hexdigest() != row['source_sha256']:
            raise ValueError('Frozen source changed: ' + row['id'])
        expected = json.loads(source)
        resolved, identity = load_config(row['template'], local)
        if identity['local_bindings_sha256'] != provenance['local_bindings_sha256']:
            raise ValueError('Local bindings changed during verification')
        if encoded(expected) != encoded(resolved):
            raise ValueError('Resolved configuration differs: ' + row['id'])
        bind(row['source'], row['source_sha256'])
        bind(row['template'], identity['template_sha256'])
        reports.append(dict(id=row['id'], passed=True, template=str(Path(row['template']).resolve()),
                            source=str(Path(row['source']).resolve()), provenance=identity))
    for path, digest in sources.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('Source changed during verification: ' + path)
    return dict(schema='portable-config-verification-result-v1', passed=True,
        provenance=provenance, sources=sources, entries=reports, training_dispatched=False,
        scope='Exact configuration values and pinned source identity only; not runtime admission or asset validation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', default=os.environ.get('JEV_LOCAL_CONFIG','.local/config/macros.json'))
    args = parser.parse_args()
    print(json.dumps(verify(args.config,args.local),indent=2))

"""Read-only template equivalence check against an immutable training package."""
import argparse
import json
from pathlib import Path

from multimodal_jev.local_config import load_config
from tools.reports.committed_signal_report import read, sha


def verify(templates, local, source):
    templates, source = Path(templates), Path(source)
    provenance = read(templates / 'SOURCE.json')
    if provenance['schema'] != 'portable-training-template-source-v1':
        raise ValueError('Unsupported source identity schema')
    results = []
    for template, name in [('context-grpo-actor-v1.json', 'config.json'),
                           ('context-grpo-service-v1.json', 'service-config.json')]:
        if sha(source / name) != provenance['source_sha256'][name]:
            raise ValueError('Frozen source changed: ' + name)
        resolved, identity = load_config(templates / template, local)
        expected = read(source / name)
        if name == 'config.json':
            expected['training_dispatch_allowed'] = False
        if resolved != expected:
            raise ValueError('Resolved template differs from baseline: ' + template)
        results.append(dict(template=template, passed=True, **identity))
    return dict(passed=True, templates=results, training_dispatched=False,
                scope='Local configuration equivalence; no runtime or training admission')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--templates', default='configs/portable/training')
    parser.add_argument('--local', required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.templates, args.local, args.source), indent=2))


if __name__ == '__main__':
    main()

"""Materialize shared JSON templates for existing config-consuming entrypoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.committed_signal_report import read, sha, write


def materialization_identity(provenance, local):
    """Bind only the referenced macro dependency closure, including aliases."""
    raw = Path(local).read_bytes()
    if hashlib.sha256(raw).hexdigest() != provenance['local_bindings_sha256']:
        raise ValueError('Local bindings changed during resolution')
    macros = json.loads(raw)['macros']
    selected = {name: macros[name] for name in provenance['macros_used']}
    digest = hashlib.sha256(json.dumps(selected, sort_keys=True, ensure_ascii=False,
                                       allow_nan=False).encode()).hexdigest()
    return dict(schema='jev-materialization-identity-v2',
        template_sha256=provenance['template_sha256'],
        resolved_semantic_sha256=provenance['resolved_semantic_sha256'],
        used_bindings_sha256=digest, macros_used=provenance['macros_used'])


def materialize(template, local, output):
    config, provenance = load_config(template, local)
    identity = materialization_identity(provenance, local)
    output = Path(output).resolve()
    check_paths({'resolved_config': output.parent})
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = output.with_name(output.name + '.provenance.json')
    # Exclusive creation of this small owner file prevents competing publishers.
    import fcntl
    with output.with_name(output.name + '.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists() or receipt.exists():
            if not output.exists() or not receipt.exists():
                raise ValueError('Different or partial existing config; use a new output version')
            previous = read(receipt)
            # Historical v1 receipts remain strict and are never rewritten.
            if previous.get('schema') == 'jev-materialized-config-v2':
                valid = (previous.get('identity') == identity
                         and previous.get('output_sha256') == sha(output))
            else:
                valid = previous == dict(provenance, output_sha256=sha(output))
            if not valid or read(output) != config:
                raise ValueError('Different or partial existing config; use a new output version')
            return dict(status='already_resolved', output=str(output))
        write(output, config)
        write(receipt, dict(schema='jev-materialized-config-v2', identity=identity,
                           original_provenance=provenance, output_sha256=sha(output)))
    return dict(status='resolved', output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', default=os.environ.get('JEV_LOCAL_CONFIG', '.local/config/macros.json'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.config, args.local, args.output)))


if __name__ == '__main__':
    main()

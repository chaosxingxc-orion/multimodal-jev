#!/usr/bin/env python3
"""Require identical request/prompt/tokens for the original and ASR reuse control."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical


def compare(original, reused):
    runs = [Path(original), Path(reused)]
    identities, predictions = [], []
    for run in runs:
        identity = json.loads((run / 'identity.json').read_text())
        result = json.loads((run / 'result.json').read_text())
        terminal = json.loads((run / 'terminal.json').read_text())
        if terminal['exit_code'] != 0 or result['status'] != 'passed':
            raise ValueError('two completed runs required')
        if result['predictions_sha256'] != file_hash(run / 'predictions.jsonl'):
            raise ValueError('prediction hashes changed')
        rows = list(map(json.loads, (run / 'predictions.jsonl').read_text().splitlines()))
        records = {(r['example_id'], r['replicate']): r for r in rows}
        selection = json.loads((run / 'selection.json').read_text())['ids']
        expected = {(uid, i) for uid in selection for i in range(identity['config']['sampling']['n'])}
        if not expected or set(records) != expected or len(records) != len(rows):
            raise ValueError('complete unique prediction coverage required')
        identities.append(identity); predictions.append(records)
    keys = ('route', 'parameters_frozen', 'requests', 'split', 'source_groups', 'device',
            'dtype', 'seed', 'max_context_tokens', 'max_new_tokens', 'asr_max_new_tokens', 'sampling')
    if any(identities[0]['config'][k] != identities[1]['config'][k] for k in keys):
        raise ValueError('control configuration changed')
    for k in ('models', 'requests_sha256', 'instruction_sha256'):
        if identities[0][k] != identities[1][k]: raise ValueError('control identity changed')
    if set(predictions[0]) != set(predictions[1]): raise ValueError('control selection changed')
    fields = ('request_sha256', 'prompt_sha256', 'input_tokens', 'token_ids', 'raw_text',
              'format_valid', 'schema_valid', 'length_limited')
    mismatches = {field: sum(predictions[0][key][field] != predictions[1][key][field]
                             for key in predictions[0]) for field in fields}
    return {'schema': 'jev-frozen-reuse-control-v1',
            'passed': not any(mismatches.values()), 'predictions': len(predictions[0]),
            'mismatch_counts': mismatches,
            'runs': [str(p) for p in runs],
            'result_sha256': [file_hash(p / 'result.json') for p in runs],
            'scope': 'exact frozen output equivalence on this selection; no efficacy claim'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--reused', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = compare(args.original, args.reused)
    with args.output.open('xb') as handle: handle.write(canonical(report))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)

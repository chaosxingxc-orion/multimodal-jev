#!/usr/bin/env python3
"""Reference contract for terminal request coverage; no model runner or admission.

Synthetic diagnostic receipts only. A production scorer must independently
authenticate raw backend output, input/model identities, truth and this mapping.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


COORDINATE_FIELDS = ('request_id', 'recording_id', 'panel_id', 'arm', 'transform')
TERMINAL = ('completed', 'failed')


def coordinate(row):
    if not isinstance(row, dict) or any(type(row.get(k)) is not str or not row[k]
                                        for k in COORDINATE_FIELDS):
        raise ValueError('complete string request coordinate required')
    return tuple(row[k] for k in COORDINATE_FIELDS)


def validate_terminal_coverage(expected, receipts):
    """Failures retain their coordinate; absence/pending/duplicates block scoring."""
    roster = {}
    for row in expected:
        key = coordinate(row)
        if key[0] in roster:
            raise ValueError('duplicate scheduled request ID')
        roster[key[0]] = key
    if not roster:
        raise ValueError('nonempty fixed request population required')
    seen = set(); counts = Counter()
    for receipt in receipts:
        key = coordinate(receipt)
        if key[0] not in roster or key != roster[key[0]]:
            raise ValueError('unknown or mismatched request coordinate')
        if key[0] in seen:
            raise ValueError('duplicate terminal receipt')
        if receipt.get('status') not in TERMINAL:
            raise ValueError('pending or unknown status is not a terminal failure')
        if receipt['status'] == 'failed':
            if type(receipt.get('error')) is not str or not receipt['error'].strip():
                raise ValueError('explicit failure evidence required')
        seen.add(key[0]); counts[receipt['status']] += 1
    if seen != set(roster):
        raise ValueError('incomplete terminal coverage; do not fabricate failures')
    return dict(requests=len(roster), completed=counts['completed'], failed=counts['failed'],
                terminal_coverage=1.0)


def decode_spans(receipt, *, final_position, native_symbols, process_symbols):
    """Normalize registered output symbols, without parsing raw text or guessing.

Positions are JSON string keys. Missing active positions invalidate that node;
unknown positions or a malformed map invalidate the entire response envelope.
An explicitly failed request has no usable partial output.
"""
    if receipt.get('status') not in TERMINAL:
        raise ValueError('terminal receipt required')
    positions = [final_position, *process_symbols]
    if (len(positions) != len(set(positions)) or
            any(type(p) is not str or not p.isdecimal() for p in positions)):
        raise ValueError('distinct registered span positions required')
    if (not native_symbols or any(type(k) is not str or type(v) is not str
                                  for k, v in native_symbols.items()) or
            len(set(native_symbols.values())) != len(native_symbols)):
        raise ValueError('one-to-one native symbol map required')
    if any(not m or any(type(k) is not str or type(v) is not bool for k, v in m.items())
           or len(m) != 2 or set(m.values()) != {False, True}
           for m in process_symbols.values()):
        raise ValueError('two registered boolean symbols required')
    spans = receipt.get('spans')
    envelope_valid = (receipt['status'] == 'completed' and type(spans) is dict
                      and set(spans) <= set(positions))
    def selected(position, mapping):
        raw = spans.get(position) if envelope_valid else None
        return mapping.get(raw) if type(raw) is str else None
    return dict(final=selected(final_position, native_symbols),
                process={p: selected(p, m) for p, m in process_symbols.items()},
                envelope_valid=envelope_valid)


def audit(request_file, output):
    if output.exists():
        raise FileExistsError(output)
    rows = [json.loads(line) for line in request_file.read_text().splitlines()]
    # Every row is explicitly synthetic; these are never stored as real predictions.
    receipts = (dict(row, status='failed', error='synthetic arithmetic fixture') for row in rows)
    full = validate_terminal_coverage(rows, receipts)
    rejected = {}
    samples = [dict(r, status='completed') for r in rows[:2]]
    mutations = {'missing':samples[:1], 'duplicate':[samples[0],samples[0]],
                 'unknown':[samples[0],dict(samples[1],request_id='foreign')],
                 'wrong_panel':[samples[0],dict(samples[1],panel_id='other')],
                 'pending':[samples[0],dict(samples[1],status='running')]}
    for name, values in mutations.items():
        try:
            validate_terminal_coverage(rows[:2], values)
        except ValueError as exc:
            rejected[name] = str(exc)
        else:
            raise AssertionError('invalid fixture was admitted: '+name)
    result = dict(schema='semantic-prediction-ledger-reference-audit-v1',
        request_file=str(request_file), request_file_sha256=hashlib.sha256(request_file.read_bytes()).hexdigest(),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        full_synthetic_failure_population=full, rejected_invalid_fixtures=rejected,
        semantic_admission=0, training_dispatch_allowed=False, evaluation_dispatch_allowed=False,
        scope='Synthetic terminal coverage reference on the real request roster; no actual inference, prediction receipts, raw parser authentication or scorer acceptance.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--requests',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(audit(a.requests,a.output)))

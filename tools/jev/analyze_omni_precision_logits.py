#!/usr/bin/env python3
"""Audit saved diagnostic logits without loading weights or replaying inference.

Artifact completeness is deliberately separate from container/source acceptance
and from the registered actual-verl probability parity gate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


CELLS = ['reference', 'autocast', 'fp32_features', 'audio_padding', 'text_padding']
FLAGS = [[False, False, False, False], [True, False, False, False],
         [True, True, False, False], [True, True, True, False],
         [True, True, True, True]]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, recorded, name):
    import torch
    other = torch.as_tensor(recorded, dtype=torch.float64)
    if actual.shape != other.shape or not torch.allclose(
            actual.double(), other, rtol=0, atol=1e-7):
        raise ValueError(f'{name} differs from saved logits')


def inspect_record(run, record, identity, previous=None):
    import torch
    row, cell = record['row'], record['cell']
    if (type(row) is not int or not 0 <= row < len(identity['keys'])
            or cell not in CELLS or record['key'] != identity['keys'][row]):
        raise ValueError('record identity differs')
    path = run / f'row-{row:02d}-{cell}-logits.pt'
    if digest(path) != record['logits_sha256']:
        raise ValueError('logits hash differs')
    logits = torch.load(path, map_location='cpu', weights_only=True)
    ids = identity['response_ids']
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
            or logits.shape[0] != len(ids) or not torch.isfinite(logits).all()
            or logits.dtype != torch.bfloat16 or str(logits.dtype) != record['logits_dtype']
            or any(type(i) is not int or not 0 <= i < logits.shape[1] for i in ids)):
        raise ValueError('invalid finite BF16 response logits')
    labels = torch.tensor(ids).unsqueeze(-1)
    logp = logits.float().log_softmax(-1)
    high = logp.gather(-1, labels).squeeze(-1)
    low = logits.log_softmax(-1).gather(-1, labels).squeeze(-1).float()
    close(high, record['fp32_log_probs'], 'FP32 log probabilities')
    close(low, record['native_dtype_log_probs'], 'native log probabilities')
    close((high-low).abs().max(), record['normalization_max_error'], 'normalization error')
    if not math.isfinite(record['seconds']) or record['seconds'] <= 0:
        raise ValueError('invalid timing')
    summary = dict(row=row, key=record['key'], cell=cell,
                   fp32_log_probs=high.tolist(), native_dtype_log_probs=low.tolist(),
                   normalization_max_error=float((high-low).abs().max()),
                   logits_sha256=record['logits_sha256'], seconds=record['seconds'])
    if previous is None:
        if cell != 'reference' or record['predecessor_max_error'] is not None:
            raise ValueError('missing predecessor')
    else:
        if logits.shape != previous.shape:
            raise ValueError('vocabulary shape changed')
        prior_logp = previous.float().log_softmax(-1)
        delta = (high-prior_logp.gather(-1, labels).squeeze(-1)).abs().max()
        close(delta, record['predecessor_max_error'], 'predecessor response error')
        summary.update(predecessor_response_max_error=float(delta),
                       predecessor_full_logits_max_error=float((logits.float()-previous.float()).abs().max()),
                       predecessor_full_logprobs_max_error=float((logp-prior_logp).abs().max()),
                       predecessor_logits_equal=bool(torch.equal(logits, previous)))
    return summary, logits


def analyze(run, identity_hash):
    identity_path = run / 'identity.json'
    if digest(identity_path) != identity_hash:
        raise ValueError('registered identity hash differs')
    identity = json.loads(identity_path.read_text())
    if (len(identity['keys']) != 6 or len({tuple(k) for k in identity['keys']}) != 6
            or identity['cells'] != [[n, *f] for n, f in zip(CELLS, FLAGS)]
            or identity['targets_read'] is not False or identity['weight_updates'] != 0
            or identity['preflight_only'] is not False or len(identity['response_ids']) != 2):
        raise ValueError('complete registered TRAIN6 five-condition diagnostic required')
    expected = [(i, c) for i in range(6) for c in CELLS]
    names = [f'row-{i:02d}-{c}.json' for i, c in expected]
    # Snapshot completed record names once; a concurrent writer may add more.
    present = {p.name for p in run.glob('row-*.json')}
    if present != set(names[:len(present)]):
        raise ValueError('records are not a contiguous diagnostic prefix')
    summaries, records, hashes = [], [], {}
    previous = None
    for (row, cell), name in zip(expected, names):
        if name not in present:
            break
        path = run / name
        record = json.loads(path.read_text())
        if (record['row'], record['cell']) != (row, cell):
            raise ValueError('record filename identity differs')
        summary, previous = inspect_record(run, record, identity,
                                           None if cell == 'reference' else previous)
        summaries.append(summary); records.append(record); hashes[name] = digest(path)
    complete = len(records) == 30
    # This flag describes only this snapshot of artifacts, never job success.
    return dict(schema='omni-precision-logit-audit-v1',
                status='artifacts_complete' if complete else 'partial_artifacts',
                identity_sha256=identity_hash, completed_cells=len(records), expected_cells=30,
                record_sha256=hashes, records=summaries,
                inference_replayed=False, parity_accepted=False,
                scope='Saved logits only; no container/source/weight acceptance or actual-verl parity verdict.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--identity-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run, args.identity_sha256)
    # Never overwrite an earlier observation or mutate the original run.
    if args.output.resolve().is_relative_to(args.run.resolve()):
        raise ValueError('audit output must be outside the immutable run')
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')

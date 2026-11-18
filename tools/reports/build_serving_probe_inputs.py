"""Materialize a fixed, label-free serving record set from the TRAIN audio cache.

Dev-state ``v0.0.1`` inputs builder. It reads the already-encoded TRAIN context
cache (one ``.pt`` per request identity), takes the first ``--rows`` entries in
lexicographic cache-key order, and writes ``inputs.pt`` plus a provenance
manifest. Selection is by cache key only: no gold label, probability, length or
reward is opened, so this cannot be mistaken for dataset sampling for training.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(cache_dir, rows_wanted, output):
    import torch
    from tools.reports.context_bulk_rpc_candidate import validate_rows
    cache_dir = Path(cache_dir)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = sorted(cache_dir.glob('*.pt'))
    if len(files) < rows_wanted:
        raise ValueError('cache population smaller than requested rows')
    rows, receipts, lengths = [], [], []
    for path in files:
        if len(rows) == rows_wanted:
            break
        entry = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(entry, dict) or set(entry) != {'binding', 'inputs', 'positions',
                                                          'audio_receipt', 'tensor_hashes'}:
            continue
        row = {'inputs': entry['inputs'], 'positions': entry['positions']}
        try:
            validate_rows([row])
        except ValueError:
            continue
        rows.append(row)
        lengths.append(int(row['inputs']['input_ids'].shape[-1]))
        receipts.append(dict(cache_key=path.stem, cache_sha256=sha(path),
            audio_sha256=entry['binding']['audio_sha256']))
    if len(rows) != rows_wanted:
        raise ValueError('insufficient valid cache entries')
    payload = output / 'inputs.pt'
    torch.save(rows, payload)
    lengths.sort()
    result = dict(schema='jev-serving-probe-inputs-v1', status='label_free_records_materialized',
        rows=len(rows), selection='first N encoded TRAIN cache entries in lexicographic cache-key order; label-free',
        cache_dir=str(cache_dir), cache_population=len(files), input_sha256=sha(payload),
        tokens=dict(min=lengths[0], max=lengths[-1],
            mean=sum(lengths) / len(lengths), p50=lengths[len(lengths) // 2]),
        receipts=receipts, training_dispatch_allowed=False,
        scope='serving timing records; not a training/selection sample')
    (output / 'MANIFEST.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--rows', type=int, required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(build(a.cache_dir, a.rows, a.output), indent=2))

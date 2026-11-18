"""Read-only inventory of archived fixed-reference records against input receipts.

This proves archived input coverage, not numerical equivalence to a new runtime.
It never copies a cache, launches training, or grants cross-runtime reuse.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct

from multimodal_jev.local_config import load_config
from tools.jev.context_cache_receipts import parse_stream


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_record(payload, name, reference, candidates):
    if (set(payload) != {'reference', 'record', 'record_sha256'}
            or payload['reference'] != reference):
        raise ValueError('reference identity differs')
    record = payload['record']
    if (set(record) != {'request_id', 'request_identity', 'prob', 'policy_id'}
            or digest(record) != payload['record_sha256']
            or record['policy_id'] != reference['policy_id']):
        raise ValueError('reference record identity differs')
    identity = record['request_identity']
    if (set(identity) != {'policy_id', 'inputs', 'positions'}
            or identity['policy_id'] != reference['policy_id']
            or digest(identity) != record['request_id']
            or name != record['request_id'] + '.json'):
        raise ValueError('request identity differs')
    positions = identity['positions']
    if (len(positions) != candidates or any(type(v) is not int or v < 0 for v in positions)
            or positions != sorted(set(positions))):
        raise ValueError('ordered marker positions required')
    probs = record['prob']
    if (len(probs) != candidates or any(type(v) not in (float, int)
            or not math.isfinite(v) or not 0 <= v <= 1 for v in probs)):
        raise ValueError('finite candidate probabilities required')
    tensors = dict(identity['inputs'])
    if set(tensors) != {'input_ids', 'attention_mask', 'input_features', 'input_features_mask'}:
        raise ValueError('complete label-free input required')
    if positions[-1] >= tensors['input_ids']['shape'][1]:
        raise ValueError('marker outside input')
    tensors['positions'] = dict(shape=[1, candidates], dtype='torch.int64',
        sha256=hashlib.sha256(struct.pack('<' + 'q' * candidates, *positions)).hexdigest())
    return digest(tensors), sum(v in (0, 1) for v in probs)


def run(c):
    paths = {key: Path(c[key]) for key in ('transport', 'receipts', 'input_proof')}
    for key, path in paths.items():
        if sha(path) != c[key + '_sha256']:
            raise ValueError('source changed: ' + key)
    wire = json.loads(paths['transport'].read_text())
    state = wire['state']
    if (wire['status'] != 'transport_complete_independent_audit_required'
            or state['Status'] != 'exited' or state['ExitCode'] != 0 or state['OOMKilled']
            or any(wire[k] for k in ('bad_pins', 'bad_package', 'launch_binding_mismatches'))):
        raise ValueError('successful bound transport required')
    proof = json.loads(paths['input_proof'].read_text())
    if (proof['independent_receipts_sha256'] != c['receipts_sha256']
            or proof['state']['Status'] != 'exited' or proof['state']['ExitCode'] != 0
            or proof['state']['OOMKilled']):
        raise ValueError('independent input audit required')
    stream, _ = parse_stream(paths['receipts'].read_text())
    rows = [x['value'] for x in stream[1:-1]]
    if (len(rows) != c['recordings'] or len({r['id'] for r in rows}) != c['recordings']
            or proof['manifest']['recordings'] != c['recordings']):
        raise ValueError('complete independent input population required')
    expected = Counter(digest(r['tensors']) for r in rows)
    root = Path(c['collection'])

    def bound(relative):
        path = root / relative
        if (path.is_symlink() or any(p.is_symlink() for p in path.parents)
                or sha(path) != wire['files'].get(relative)):
            raise ValueError('changed/unbound collection file: ' + relative)
        return path

    config = json.loads(bound('config.json').read_text())
    if proof['manifest']['release_sha256'] != config['release_sha256']:
        raise ValueError('input release differs')
    reference = dict(schema='context-fixed-reference-cache-v1', policy_id=config['source_policy_id'],
                     base_id=config['base_id'], contract=config['source_contract'])
    namespace = digest(reference)
    entries = sorted(k for k in wire['files'] if k.startswith('reference-cache/'))
    actual = Counter(); total_bytes = saturated = 0
    for entry in entries:
        relative = Path(entry)
        if (len(relative.parts) != 3 or relative.parts[1] != namespace
                or relative.suffix != '.json'):
            raise ValueError('unexpected cache namespace/file')
        path = bound(entry)
        payload = json.loads(path.read_text())
        key, endpoints = validate_record(payload, path.name, reference, c['candidates'])
        actual[key] += 1; saturated += endpoints; total_bytes += path.stat().st_size
    # Dataset entries in this release have distinct complete tensor identities.
    if any(v != 1 for v in expected.values()) or actual != expected:
        raise ValueError('reference cache does not cover exactly the independent input population')
    return dict(schema='context-reference-inventory-v1', archived_input_coverage_passed=True,
        recordings=len(rows), cache_entries=len(entries), candidates=c['candidates'],
        archived_json_bytes=total_bytes, exact_zero_or_one_probabilities=saturated,
        reference_policy_id=reference['policy_id'], namespace=namespace,
        source_image=wire['image'], source_container=wire['container_id'],
        sources={k: c[k + '_sha256'] for k in paths},
        cross_runtime_reuse_allowed=False, cache_copied=False,
        limitation='Archive hashes and exact input coverage only; no new-runtime numerical equivalence, raw-logit saturation coverage, cache publication, or throughput measurement.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--local', required=True)
    args = parser.parse_args(); config, provenance = load_config(args.config, args.local)
    result = run(config); result['config_provenance'] = provenance
    result['implementation_sha256'] = sha(__file__)
    with Path(config['output']).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(result, indent=2))

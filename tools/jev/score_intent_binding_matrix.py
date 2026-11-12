#!/usr/bin/env python3
"""Authenticate complete paired gates, then report all label-free interventions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.span_binding_metrics import compare, probabilities, summarize
from multimodal_jev.span_candidate_intervention import VARIANTS
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read, completed


def validate_records(inputs, hf, vl, base_id, adapter_id):
    rows = {r['id']: r for r in inputs['records']}
    if len(rows) != len(inputs['records']): raise ValueError('unique CPU inputs required')
    backends = []
    for name, result in [('hf', hf), ('vllm', vl)]:
        indexed = {r['id']: r for r in result['records']}
        if (not result['passed'] or result['backend'] != name or result['base_id'] != base_id
                or result['adapter_id'] != adapter_id or indexed.keys() != rows.keys()
                or len(indexed) != len(result['records'])): raise ValueError('complete identical policy population required')
        backends.append(indexed)
    worst = 0.; low = high = 1.
    for uid, row in rows.items():
        left, right = (b[uid] for b in backends); identity = left['request_identity']
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if right['request_identity'] != identity or left['request_id'] != digest or right['request_id'] != digest:
            raise ValueError('backend request identity differs')
        if (identity['version'] != 'jev-span-request-identity-v1' or identity['request'] != row['request']
                or identity['policy_id'] != base_id+':'+adapter_id
                or identity['sampling'] != dict(temperature=1.0, policy='independent_finite_categorical')
                or identity['audio_processing'] != row['audio_processing']
                or identity['positions'] != row['positions']
                or identity['candidate_ids'] != [m['token_ids'] for m in row['candidate_mapping']]
                or any(identity['inputs'][k] != row[k] for k in ('input_ids', 'input_features', 'input_features_mask'))):
            raise ValueError('backend input differs from CPU evidence')
        a, b = left['candidate_logprobs'], right['candidate_logprobs']
        if len(a) != len(row['candidate_mapping']) or len(b) != len(a): raise ValueError('Span support differs')
        lower = upper = 0.
        for x, y, mapping in zip(a, b, row['candidate_mapping'], strict=True):
            probabilities(x); probabilities(y)
            if len(x) != len(y) or len(x) != len(mapping['values']): raise ValueError('candidate support differs')
            delta = [u-v for u,v in zip(x,y,strict=True)]
            worst = max(worst, max(map(abs, delta))); lower += min(delta); upper += max(delta)
        if worst > .02 or lower < math.log(.98) or upper > math.log(1.02): raise ValueError('full-support parity failed')
        low = min(low, math.exp(lower)); high = max(high, math.exp(upper))
    return backends[1], dict(requests=len(rows), maximum_logprob_error=worst, joint_ratio_range=[low,high])


def authenticate(entry, protocol_path, digest, protocol):
    gate = Path(entry['root']); cfg = read(gate/'config.json'); launch = read(gate/'launch.json')
    if (launch['image'] != protocol['image'] or launch['physical_devices'] != [4]
            or not completed(launch['container_id'], protocol['image'])): raise ValueError('completed exact gate required')
    check_pins({str(gate/'config.json'): launch['config_sha256'], str(gate/'source-manifest.json'): launch['source_manifest_sha256']})
    check_pins(cfg['pins']); check_pins({str(gate/'source'/p):h for p,h in read(gate/'source-manifest.json').items()})
    if (cfg['pins'].get(str(protocol_path)) != digest or cfg['inputs'] != protocol['inputs']
            or cfg['pins'].get(protocol['inputs']) != protocol['inputs_sha256']
            or cfg['base_id'] != protocol['base_id'] or cfg['adapter_id'] != entry['adapter_id']
            or cfg['runtime_profile'] != protocol['runtime_profile']
            or cfg['population'] != dict(recordings=89, requests=712)): raise ValueError('fixed diagnostic protocol required')
    for name in ('terminal.json', 'hf/terminal.json', 'vllm/terminal.json'):
        if read(gate/'run'/name)['exit_code'] != 0: raise ValueError('gate phase failed')
    ready = read(gate/'run/READY.json')
    if (not ready['passed'] or ready['requests'] != 712 or ready['adapter_id'] != entry['adapter_id']
            or ready['base_id'] != protocol['base_id']
            or ready['result_sha256'] != file_hash(gate/'run/vllm/result.json')): raise ValueError('gate readiness differs')
    return validate_records(read(protocol['inputs']), read(gate/'run/hf/result.json'), read(gate/'run/vllm/result.json'),
                            protocol['base_id'], entry['adapter_id'])


def run(path, digest):
    path = Path(path)
    if file_hash(path) != digest: raise ValueError('protocol changed')
    protocol = read(path)
    if (protocol['schema'] != 'intent-binding-diagnostic-v1' or protocol['variants'] != list(VARIANTS)
            or [e['key'] for e in protocol['checkpoints']] != ['step0','step4']): raise ValueError('fixed matrix required')
    check_pins({protocol['inputs']:protocol['inputs_sha256']})
    inputs = read(protocol['inputs']); reports = {}
    for entry in protocol['checkpoints']:
        records, audit = authenticate(entry, path, digest, protocol); groups = {}
        for row in inputs['records']:
            if row['variant'] == 'native': continue
            ref = records[row['recording_id']+'::'+row['arm']+'::native']; changed = records[row['id']]
            for j,q in enumerate(row['request']['questions']):
                key = row['arm']+'::'+row['variant']+'::'+q['id']
                groups.setdefault(key, []).append(compare(ref['candidate_logprobs'][j], changed['candidate_logprobs'][j], row['binding'][q['id']]))
        reports[entry['key']] = dict(audit=audit, comparisons={k:summarize(v) for k,v in groups.items()})
    return dict(protocol_sha256=digest, reports=reports, scope='All fixed label-free paired views; display order and symbol binding confounded. No task efficacy or audio causality claim.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--protocol-sha256',required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError('immutable diagnostic exists')
    atomic_json(a.output,run(a.protocol,a.protocol_sha256))

import copy
import hashlib
import json
import math
import string

import pytest

from test_fsc_transfer_metrics import fixture
from multimodal_jev.fsc_span_transfer import compile_request
from multimodal_jev.fsc_backend_records import normalize_gate
from multimodal_jev.fsc_transfer_metrics import score_policy


def digest(row):
    row['request_id'] = hashlib.sha256(json.dumps(row['request_identity'], sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def inputs():
    task, roster, targets, predictions = fixture()
    rows = []; outputs = []
    for pred in predictions:
        _, template, variant = pred['id'].split('::')
        req, binding = compile_request(roster[0]['audio'], task, template, variant=variant, seed=317)
        mapping = [dict(values=list(q.values), token_ids=list(range(len(q.values))),
                       symbols=list((string.ascii_uppercase+string.ascii_lowercase)[:len(q.values)]))
                   for q in req.questions]
        features = {k: {'sha256': k} for k in ('input_ids', 'input_features', 'input_features_mask')}
        row = dict(id=pred['id'], request=pred['request_identity']['request'], binding=binding,
                   audio_sha256='hash', audio_processing={'receipt': 'test'},
                   positions=[list(range(len(req.questions)))], candidate_mapping=mapping, **features)
        identity = dict(version='jev-span-request-identity-v1', request=row['request'],
            policy_id='base:adapter', sampling={'temperature': 1.0, 'policy': 'independent_finite_categorical'},
            audio_processing=row['audio_processing'], positions=row['positions'],
            candidate_ids=[m['token_ids'] for m in mapping], inputs=features)
        out = dict(id=row['id'], request_identity=copy.deepcopy(identity), candidate_logprobs=pred['candidate_logprobs'])
        digest(out); rows.append(row); outputs.append(out)
    artifact = dict(passed=True, recordings=1, requests=6, records=rows)
    hf = dict(passed=True, backend='hf', base_id='base', adapter_id='adapter', records=outputs)
    vl = copy.deepcopy(hf); vl['backend'] = 'vllm'
    return task, roster, targets, artifact, hf, vl


def run(data):
    task, roster, _, artifact, hf, vl = data
    return normalize_gate(task, roster, artifact, hf, vl, base_id='base', adapter_id='adapter')


def test_backend_output_connects_to_semantic_metrics():
    data = inputs(); normalized = run(data)
    scored = score_policy(data[0], data[1], data[2], normalized['records'], policy_id='base:adapter')
    assert normalized['audit']['joint_ratio_range'] == [1., 1.]
    assert all(c['final_accuracy'] == 1 for c in scored['cells'].values())


@pytest.mark.parametrize('bad', ['missing', 'duplicate', 'policy', 'request', 'digest', 'audio',
                                'support', 'normalization', 'mapping', 'candidate_ids', 'features', 'parity'])
def test_rejects_mismatched_backend_or_cpu_evidence(bad):
    data = inputs(); _, _, _, artifact, hf, vl = data
    row = vl['records'][0]
    if bad == 'missing': vl['records'].pop()
    if bad == 'duplicate': vl['records'].append(row)
    if bad == 'policy': vl['adapter_id'] = 'wrong'
    if bad == 'request': row['request_identity']['request']['requirements'] = ['wrong']
    if bad == 'digest': row['request_id'] = 'wrong'
    if bad == 'audio': artifact['records'][0]['audio_sha256'] = 'wrong'
    if bad == 'support': row['candidate_logprobs'][0].pop()
    if bad == 'normalization': row['candidate_logprobs'][0][0] = 1.
    if bad == 'mapping': artifact['records'][0]['candidate_mapping'][0]['values'].reverse()
    if bad in ('candidate_ids', 'features'):
        for backend in (hf, vl):
            r = backend['records'][0]
            if bad == 'candidate_ids': r['request_identity']['candidate_ids'][0].reverse()
            else: r['request_identity']['inputs']['input_ids'] = {'sha256': 'wrong'}
            digest(r)
    if bad == 'parity':
        p = row['candidate_logprobs'][0]; p[0], p[1] = p[1], p[0]
        row['parity'] = {'passed': True}  # Stored pass flags cannot override recomputation.
    with pytest.raises(ValueError): run(data)


def test_joint_ratio_catches_small_errors_accumulating_across_spans():
    data = inputs(); vl = data[-1]
    row = next(r for r in vl['records'] if '::structured::native' in r['id'])
    for i, values in enumerate(row['candidate_logprobs']):
        changed = [v + (.01 if k == 0 else -.01) for k, v in enumerate(values)]
        z = math.log(sum(map(math.exp, changed)))
        row['candidate_logprobs'][i] = [v-z for v in changed]
        assert max(abs(a-b) for a, b in zip(values, row['candidate_logprobs'][i])) < .02
    with pytest.raises(ValueError, match='parity'): run(data)

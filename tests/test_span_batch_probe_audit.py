import copy
import hashlib
import json
import math

import pytest

from multimodal_jev.span_batch_probe_audit import audit_probe
from multimodal_jev.span_parity import compare_span_policies


def fixture():
    records = []
    for i in range(32):
        identity = dict(policy_id='base:adapter', candidate_ids=[list(range(60))],
                        inputs={'input_ids': {'shape': [1, 100+i]}}, uid=str(i))
        wire = json.dumps(identity, sort_keys=True, separators=(',', ':'))
        records.append(dict(id=str(i), request_id=hashlib.sha256(wire.encode()).hexdigest(),
                            request_identity=identity, tokens=100+i, frames=200+i))
    probs = [[[-math.log(60)] * 60] for _ in range(32)]
    measures = []
    for size in [1, 2, 4, 8, 16, 32]:
        measures.append(dict(microbatch=size, backbone_calls=32//size, seconds=2.,
            requests_per_second=16., maximum_allocated_bytes=100, maximum_reserved_bytes=200,
            candidate_logprobs=copy.deepcopy(probs), passed=True,
            parity=[compare_span_policies(p, p) for p in probs]))
    summary = dict(results=[{k:v for k,v in r.items() if k not in ('parity','candidate_logprobs')}
                            for r in measures], passed=True, training_promotion=False)
    return dict(records=records), dict(candidate_logprobs=probs), measures, summary


def test_recomputes_complete_ladder():
    result = audit_probe(*fixture(), policy_id='base:adapter')
    assert result['passed'] and result['requests'] == 32


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'identity', 'support', 'probability',
                                  'calls', 'timing', 'summary', 'promotion', 'order'])
def test_rejects_incomplete_or_changed_evidence(fault):
    inputs, serial, rows, summary = fixture()
    if fault == 'missing': rows[0]['candidate_logprobs'].pop()
    if fault == 'duplicate': inputs['records'][1] = inputs['records'][0]
    if fault == 'identity': inputs['records'][0]['request_identity']['uid'] = 'changed'
    if fault == 'support': serial['candidate_logprobs'][0][0].pop()
    if fault == 'probability': rows[0]['candidate_logprobs'][0][0][0] += 1
    if fault == 'calls': rows[-1]['backbone_calls'] = 32
    if fault == 'timing': rows[0]['seconds'] = float('nan')
    if fault == 'summary': summary['results'][0]['seconds'] = 99
    if fault == 'promotion': summary['training_promotion'] = True
    if fault == 'order': rows.reverse()
    with pytest.raises(ValueError): audit_probe(inputs, serial, rows, summary, policy_id='base:adapter')


def test_valid_completed_negative_result_is_auditable_but_not_passing():
    inputs, serial, rows, summary = fixture()
    rows = rows[:2]
    p = [.5] + [.5/59]*59
    rows[-1]['candidate_logprobs'][0] = [[math.log(x) for x in p]]
    rows[-1]['parity'] = [compare_span_policies(a,b) for a,b in
                         zip(serial['candidate_logprobs'],rows[-1]['candidate_logprobs'])]
    rows[-1]['passed'] = False
    summary.update(results=[{k:v for k,v in r.items() if k not in ('parity','candidate_logprobs')}
                            for r in rows], passed=False)
    result = audit_probe(inputs, serial, rows, summary, policy_id='base:adapter')
    assert result['audited'] and not result['passed']

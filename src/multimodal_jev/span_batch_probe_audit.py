"""Recompute batch-probe conclusions from complete raw probabilities.

Caller verifies container/image, source/config hashes and original file bytes.
This audit proves only the recorded inference population, never training parity.
"""
import hashlib
import json
import math

from .span_parity import compare_span_policies


def audit_probe(inputs, serial, measurements, summary, *, policy_id):
    records = inputs['records']
    if (len(records) != 32 or len({r['id'] for r in records}) != 32
            or len({r['request_id'] for r in records}) != 32):
        raise ValueError('32 distinct bound requests required')
    for row in records:
        identity = row['request_identity']
        wire = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False)
        if (hashlib.sha256(wire.encode()).hexdigest() != row['request_id']
                or identity['policy_id'] != policy_id
                or len(identity['candidate_ids']) != 1 or len(identity['candidate_ids'][0]) != 60
                or len(set(identity['candidate_ids'][0])) != 60
                or row['tokens'] != identity['inputs']['input_ids']['shape'][-1]
                or type(row['frames']) is not int or row['frames'] < 1):
            raise ValueError('request identity, policy or Final60 support differs')
    refs = serial['candidate_logprobs']
    def check(values):
        if len(values) != 32 or any(len(row) != 1 or len(row[0]) != 60 for row in values):
            raise ValueError('complete32 by Final60 probabilities required')
    check(refs)
    sizes = [r['microbatch'] for r in measurements]
    if not sizes or sizes != [1, 2, 4, 8, 16, 32][:len(sizes)]:
        raise ValueError('ordered uninterrupted capacity ladder required')
    result = []
    for index, row in enumerate(measurements):
        check(row['candidate_logprobs'])
        pairs = [compare_span_policies(a, b) for a, b in zip(refs, row['candidate_logprobs'], strict=True)]
        passed = all(p['passed'] for p in pairs)
        if (row['backbone_calls'] != 32 // row['microbatch']
                or row['passed'] is not passed or row['parity'] != pairs
                or (not passed and index != len(measurements)-1)):
            raise ValueError('raw probability verdict or forward call count differs')
        seconds = row['seconds']
        if (type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds <= 0
                or not math.isclose(row['requests_per_second'], 32 / seconds, rel_tol=1e-12)
                or any(type(row[k]) is not int or row[k] < 0 for k in
                       ('maximum_allocated_bytes', 'maximum_reserved_bytes'))
                or row['maximum_reserved_bytes'] < row['maximum_allocated_bytes']):
            raise ValueError('finite timing and valid memory observations required')
        result.append(dict(microbatch=row['microbatch'], passed=passed,
            maximum_logprob_error=max(p['maximum_logprob_error'] for p in pairs),
            argmax_changed_requests=sum(bool(p['argmax_changed_spans']) for p in pairs),
            requests_per_second=32 / seconds,
            maximum_allocated_bytes=row['maximum_allocated_bytes'],
            maximum_reserved_bytes=row['maximum_reserved_bytes']))
    expected = [{k: v for k, v in row.items() if k not in ('parity', 'candidate_logprobs')}
                for row in measurements]
    complete = len(measurements) == 6 and all(r['passed'] for r in result)
    if (summary['results'] != expected or summary['passed'] is not complete
            or summary['training_promotion'] is not False):
        raise ValueError('summary differs from raw measurements')
    return dict(passed=complete, audited=True, requests=32, results=result,
                training_promotion=False,
                scope='Raw inference probabilities independently rescored; timing/memory are recorded telemetry, not independently measured. No backward, full-data stability, FSDP or direct vLLM proof.')

import copy
import hashlib
import struct

import pytest

from tools.reports.audit_context_reference_inventory import digest, validate_record


def fixture():
    reference = {'policy_id': 'a' * 64}
    inputs = {k: {'shape': [1, 10], 'dtype': 'torch.int64', 'sha256': 'b' * 64}
              for k in ('input_ids', 'attention_mask', 'input_features', 'input_features_mask')}
    identity = dict(policy_id=reference['policy_id'], inputs=inputs, positions=[3, 8])
    record = dict(request_id=digest(identity), request_identity=identity,
                  policy_id=reference['policy_id'], prob=[0.0, 0.7])
    payload = dict(reference=reference, record=record, record_sha256=digest(record))
    return payload, record['request_id'] + '.json', reference


def test_exact_position_bytes_and_endpoint_count():
    payload, name, reference = fixture()
    tensors = copy.deepcopy(payload['record']['request_identity']['inputs'])
    tensors['positions'] = dict(shape=[1, 2], dtype='torch.int64',
        sha256=hashlib.sha256(struct.pack('<qq', 3, 8)).hexdigest())
    assert validate_record(payload, name, reference, 2) == (digest(tensors), 1)


@pytest.mark.parametrize('field,value,error', [
    ('prob', [float('inf'), 0.7], 'finite candidate'),
    ('prob', [-0.1, 0.7], 'finite candidate'),
    ('prob', [0.7], 'finite candidate'),
    ('policy_id', 'c' * 64, 'record identity'),
])
def test_resigned_bad_record_rejected(field, value, error):
    payload, name, reference = fixture()
    payload['record'][field] = value
    if field == 'prob' and value[0] == float('inf'):
        with pytest.raises(ValueError): digest(payload['record'])
        return
    payload['record_sha256'] = digest(payload['record'])
    with pytest.raises(ValueError, match=error): validate_record(payload, name, reference, 2)


def test_wrong_filename_and_reference_rejected():
    payload, name, reference = fixture()
    with pytest.raises(ValueError, match='request identity'):
        validate_record(payload, 'wrong.json', reference, 2)
    with pytest.raises(ValueError, match='reference identity'):
        validate_record(payload, name, {'policy_id': 'c' * 64}, 2)


def test_reordered_markers_rejected_even_when_resigned():
    payload, _, reference = fixture()
    record = payload['record']; record['request_identity']['positions'] = [8, 3]
    record['request_id'] = digest(record['request_identity'])
    payload['record_sha256'] = digest(record)
    with pytest.raises(ValueError, match='ordered marker'):
        validate_record(payload, record['request_id'] + '.json', reference, 2)

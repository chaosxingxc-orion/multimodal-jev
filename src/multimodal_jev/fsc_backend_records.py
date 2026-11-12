"""Validate full-support FSC backend records before metric consumption.

The caller authenticates container completion and pinned files. This module
checks their contents independently; it grants no dataset/evaluation admission.
"""
import hashlib
import json
import math
import string

from .fsc_span_transfer import compile_request
from .fsc_transfer_metrics import TEMPLATES, VARIANTS


def normalize_gate(task, roster, artifact, hf, vllm, *, base_id, adapter_id, seed=317):
    policy_id = base_id + ':' + adapter_id
    sources = {r['id']: r for r in roster}
    if not sources or len(sources) != len(roster):
        raise ValueError('distinct complete audio roster required')
    expected = {r['id'] + '::' + t + '::' + v: (r, t, v)
                for r in roster for t in TEMPLATES for v in VARIANTS}
    inputs = {r['id']: r for r in artifact['records']}
    if (artifact['passed'] is not True or artifact['recordings'] != len(roster)
            or artifact['requests'] != len(expected)
            or len(inputs) != len(artifact['records']) or inputs.keys() != expected.keys()):
        raise ValueError('complete CPU input population required')
    backends = []
    for name, result in (('hf', hf), ('vllm', vllm)):
        rows = {r['id']: r for r in result['records']}
        if (result['passed'] is not True or result['backend'] != name
                or result['base_id'] != base_id or result['adapter_id'] != adapter_id
                or len(rows) != len(result['records']) or rows.keys() != expected.keys()):
            raise ValueError('complete matching backend policy required')
        backends.append(rows)
    records = []; worst = 0.; low = high = 1.; flips = 0
    for uid, (source, template, variant) in expected.items():
        row = inputs[uid]
        request, binding = compile_request(source['audio'], task, template, variant=variant, seed=seed)
        raw = json.loads(json.dumps(request.model_input()))
        if (row['request'] != raw or row['binding'] != binding
                or row['audio_sha256'] != source['audio_sha256']):
            raise ValueError('configured input or audio differs')
        left, right = (b[uid] for b in backends)
        identity = left['request_identity']
        if right['request_identity'] != identity:
            raise ValueError('backend request identities differ')
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'),
                                          allow_nan=False).encode()).hexdigest()
        if left['request_id'] != digest or right['request_id'] != digest:
            raise ValueError('request identity digest differs')
        if (identity['version'] != 'jev-span-request-identity-v1'
                or identity['request'] != raw or identity['policy_id'] != policy_id
                or identity['sampling'] != {'temperature': 1.0, 'policy': 'independent_finite_categorical'}
                or identity['audio_processing'] != row['audio_processing']
                or identity['positions'] != row['positions']
                or identity['candidate_ids'] != [m['token_ids'] for m in row['candidate_mapping']]
                or any(identity['inputs'][k] != row[k]
                       for k in ('input_ids', 'input_features', 'input_features_mask'))):
            raise ValueError('backend evidence differs from CPU input')
        mapping = row['candidate_mapping']
        if (len(mapping) != len(request.questions)
                or any(m['values'] != list(q.values)
                       or m['symbols'] != list((string.ascii_uppercase + string.ascii_lowercase + string.digits)[:len(q.values)])
                       or len(m['token_ids']) != len(q.values) or len(set(m['token_ids'])) != len(q.values)
                       for m, q in zip(mapping, request.questions, strict=True))):
            raise ValueError('complete ordered candidate support required')
        policies = [r['candidate_logprobs'] for r in (left, right)]
        for policy in policies:
            if len(policy) != len(request.questions):
                raise ValueError('complete Span probabilities required')
            for values, question in zip(policy, request.questions, strict=True):
                if (len(values) != len(question.values)
                        or any(type(v) not in (int, float) or not math.isfinite(v) or v > 1e-6 for v in values)
                        or abs(math.log(sum(math.exp(v) for v in values))) > 1e-6):
                    raise ValueError('normalized finite full-support probabilities required')
        lower = upper = 0.
        for a, b in zip(*policies, strict=True):
            delta = [x-y for x, y in zip(a, b, strict=True)]
            worst = max(worst, max(map(abs, delta)))
            lower += min(delta); upper += max(delta)
            flips += int(max(range(len(a)), key=a.__getitem__) != max(range(len(b)), key=b.__getitem__))
        if worst > .02 or lower < math.log(.98) or upper > math.log(1.02):
            raise ValueError('independent full-support parity failed')
        low = min(low, math.exp(lower)); high = max(high, math.exp(upper))
        records.append(dict(id=uid, policy_id=policy_id, audio_sha256=source['audio_sha256'],
                            request_identity=identity, candidate_logprobs=right['candidate_logprobs']))
    return dict(records=records, audit=dict(requests=len(records), maximum_logprob_error=worst,
        joint_ratio_range=[low, high], argmax_changed_spans=flips,
        scope='Content validation only; caller must authenticate pinned artifacts, actual execution and evaluation admission.'))

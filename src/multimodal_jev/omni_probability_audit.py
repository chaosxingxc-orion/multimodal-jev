"""Complete sampled-token probability comparison, independent of task labels.

The caller authenticates runtime, weights, input features and original sampling
artifacts. This module audits aligned numeric evidence; it cannot establish that
the supplied probabilities came from the claimed model or device.
"""
import math

from .sfc_reward_batch import digest, valid_hash


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _sum(values):
    try:
        value = math.fsum(values)
    except OverflowError as exc:
        raise ValueError('nonfinite log-ratio arithmetic') from exc
    if not math.isfinite(value):
        raise ValueError('nonfinite log-ratio arithmetic')
    return value


def _key(record):
    if (type(record['example_id']) is not str or not record['example_id']
            or type(record['decision_id']) is not str or not record['decision_id']
            or type(record['replicate']) is not int or record['replicate'] < 0):
        raise ValueError('valid response identity required')
    return record['example_id'], record['decision_id'], record['replicate']


def _probabilities(tokens, values):
    if (type(tokens) is not list or not tokens
            or any(type(t) is not int or t < 0 for t in tokens)
            or type(values) is not list or len(values) != len(tokens)
            or any(not _finite(v) or v > 0 for v in values)):
        raise ValueError('complete finite nonpositive sampled-token probabilities required')


def _exp(value):
    """Keep overflow explicit and JSON finite; never clip a ratio to a safe value."""
    try:
        result = math.exp(value)
    except OverflowError:
        return dict(value=None, overflow=True, underflow=False)
    return dict(value=result, overflow=False, underflow=result == 0.0)


def _distribution(values):
    ordered = sorted(values)
    return dict(count=len(values), mean=_sum(v/len(values) for v in values),
                maximum=ordered[-1], p99_nearest_rank=ordered[math.ceil(.99*len(ordered))-1])


def compare(records, runs, *, protocol, protocol_sha256):
    """Audit every supplied run and every pair without changing old logprobs.

    Protocol and its hash must be registered before actual actor execution.
    Numeric limits are explicit: no reuse of the historical CPU 0.1 threshold.
    Each run has name and responses; each response binds the exact original
    record digest, token IDs and complete valid-token log_probs (no padding).
    """
    required = {'schema', 'records_sha256', 'policy_sha256', 'instruction_sha256',
                'run_names', 'max_abs_logprob_error', 'token_ratio_lower',
                'token_ratio_upper', 'max_between_run_logprob_error'}
    if type(records) is not list or type(runs) is not list or type(protocol) is not dict:
        raise ValueError('record/run arrays and a protocol object required')
    if (set(protocol) != required or protocol['schema'] != 'omni-probability-comparison-v1'
            or not valid_hash(protocol_sha256) or digest(protocol) != protocol_sha256
            or not valid_hash(protocol['policy_sha256'])
            or not valid_hash(protocol['instruction_sha256'])
            or digest(records) != protocol['records_sha256']):
        raise ValueError('exact registered probability protocol and original records required')
    names = protocol['run_names']
    if (type(names) is not list or len(names) < 2
            or any(type(n) is not str or not n for n in names) or len(set(names)) != len(names)):
        raise ValueError('at least two unique registered run names required')
    for key in ('max_abs_logprob_error', 'max_between_run_logprob_error'):
        if not _finite(protocol[key]) or protocol[key] < 0:
            raise ValueError('explicit finite nonnegative logprob limits required')
    lower, upper = protocol['token_ratio_lower'], protocol['token_ratio_upper']
    if not _finite(lower) or not _finite(upper) or not 0 < lower <= 1 <= upper:
        raise ValueError('finite positive token-ratio bounds containing one required')
    loglower, logupper = math.log(lower), math.log(upper)
    old = {}
    for record in records:
        key = _key(record)
        if key in old:
            raise ValueError('duplicate original response')
        if (record['policy_sha256'] != protocol['policy_sha256']
                or record['instruction_sha256'] != protocol['instruction_sha256']
                or record['finish_reason'] not in ('stop', 'length')):
            raise ValueError('original policy, instruction or finish reason differs')
        _probabilities(record['token_ids'], record['log_probs'])
        old[key] = record
    if not old or len(runs) != len(names):
        raise ValueError('nonempty complete original records and run roster required')
    rebuilt = {}
    for run in runs:
        name = run['name']
        if name not in names or name in rebuilt:
            raise ValueError('unexpected or duplicate actor run')
        mapped = {}
        for response in run['responses']:
            key = _key(response)
            if key not in old or key in mapped:
                raise ValueError('unexpected or duplicate actor response')
            original = old[key]
            if (response['record_sha256'] != digest(original)
                    or response['token_ids'] != original['token_ids']):
                raise ValueError('actor response differs from original sampled record')
            _probabilities(response['token_ids'], response['log_probs'])
            mapped[key] = response['log_probs']
        if set(mapped) != set(old):
            raise ValueError('complete actor response coverage required')
        rebuilt[name] = mapped
    reports = []
    for name in names:
        rows = []; absolute = []; deltas = []; ratio_violations = 0
        for key, original in old.items():
            # Use original JSON numbers, not FP32 old_log_probs from DataProto.
            diff = [a-b for a,b in zip(rebuilt[name][key], original['log_probs'], strict=True)]
            if any(not math.isfinite(d) for d in diff):
                raise ValueError('nonfinite log-ratio arithmetic')
            error = [abs(d) for d in diff]
            bad = sum(not loglower <= d <= logupper for d in diff)
            total = _sum(diff)
            rows.append(dict(example_id=key[0],decision_id=key[1],replicate=key[2],
                finish_reason=original['finish_reason'],tokens=len(diff),
                max_abs_logprob_error=max(error),mean_abs_logprob_error=_sum(e/len(error) for e in error),
                token_log_ratios=diff,token_ratio_bounds_violations=bad,
                response_log_ratio=total,response_ratio=_exp(total),
                passed=max(error)<=protocol['max_abs_logprob_error'] and bad==0))
            absolute.extend(error);deltas.extend(diff);ratio_violations += bad
        reports.append(dict(name=name,responses=rows,token_abs_logprob_error=_distribution(absolute),
            response_mean_abs_logprob_error=_sum(r['mean_abs_logprob_error']/len(rows) for r in rows),
            token_ratio_minimum=_exp(min(deltas)),token_ratio_maximum=_exp(max(deltas)),
            token_ratio_bounds_violations=ratio_violations,failed_responses=sum(not r['passed'] for r in rows),
            passed=all(r['passed'] for r in rows)))
    between = []
    for i, left in enumerate(names):
        for right in names[i+1:]:
            errors = []
            for key in old:
                values = [abs(a-b) for a,b in zip(rebuilt[left][key], rebuilt[right][key], strict=True)]
                errors.extend(values)
            stats = _distribution(errors)
            between.append(dict(left=left,right=right,token_abs_logprob_error=stats,
                passed=stats['maximum'] <= protocol['max_between_run_logprob_error']))
    return dict(schema='omni-probability-audit-v1',protocol_sha256=protocol_sha256,
        records_sha256=protocol['records_sha256'],runs_sha256=digest(runs),
        responses=len(old),tokens=sum(len(r['token_ids']) for r in records),
        length_limited_responses=sum(r['finish_reason']=='length' for r in records),
        runs=reports,between=between,numerical_gate_passed=all(r['passed'] for r in reports+between),
        scope='Complete valid-token numeric comparison only. Sequence ratios are descriptive, not a separate admission gate or a full-vocabulary KL estimate. Runtime/weights/features and original roster require independent authentication.')


def _read_json(path):
    import json
    from pathlib import Path

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON field: {key}')
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f'nonfinite JSON constant: {value}')

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=constant)


def main(argv=None):
    """Compare saved evidence; exit 1 on numerical failure, 2 on invalid input.

    The output is created exclusively. A failed audit never overwrites an
    existing evidence file. The CLI does not load weights or execute inference.
    """
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('records', 'runs', 'protocol', 'protocol-sha256', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    try:
        report = compare(_read_json(args.records), _read_json(args.runs),
                         protocol=_read_json(args.protocol), protocol_sha256=args.protocol_sha256)
        payload = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+'\n'
        with Path(args.output).open('x') as stream:
            stream.write(payload)
    except (ValueError, TypeError, KeyError, OverflowError, OSError) as exc:
        parser.exit(2, f'invalid probability evidence: {exc}\n')
    return 0 if report['numerical_gate_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

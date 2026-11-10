"""Synthetic numeric contracts, not evidence of Omni forward equivalence."""
from copy import deepcopy
import json

import pytest

from multimodal_jev.omni_probability_audit import compare, main
from multimodal_jev.sfc_reward_batch import digest


def fixture(values=None):
    values = values or [[-1., -2.], [-3.], [-4., -5., -6.]]
    records = [dict(example_id='audio', decision_id='calls', replicate=i,
                    policy_sha256='a'*64, instruction_sha256='b'*64,
                    token_ids=list(range(10, 10+len(row))), log_probs=row,
                    finish_reason='stop' if i == 0 else 'length')
               for i, row in enumerate(values)]
    runs = [dict(name=name, responses=[dict(
        example_id=r['example_id'], decision_id=r['decision_id'], replicate=r['replicate'],
        record_sha256=digest(r), token_ids=list(r['token_ids']), log_probs=list(r['log_probs']))
        for r in records]) for name in ('micro1', 'micro2')]
    protocol = dict(schema='omni-probability-comparison-v1', records_sha256=digest(records),
                    policy_sha256='a'*64, instruction_sha256='b'*64,
                    run_names=['micro1', 'micro2'], max_abs_logprob_error=.25,
                    token_ratio_lower=.8, token_ratio_upper=1.25,
                    max_between_run_logprob_error=.1)
    return records, runs, protocol


def audit(args):
    records, runs, protocol = args
    return compare(records, runs, protocol=protocol, protocol_sha256=digest(protocol))


def test_complete_keyed_coverage_and_no_mutation():
    args = fixture()
    args[1].reverse()
    args[1][0]['responses'].reverse()
    before = deepcopy(args)
    result = audit(args)
    assert args == before
    assert result['numerical_gate_passed']
    assert result['responses'] == 3 and result['tokens'] == 6
    assert result['length_limited_responses'] == 2
    assert [r['name'] for r in result['runs']] == ['micro1', 'micro2']
    assert result['runs'][0]['token_ratio_minimum']['value'] == 1
    assert result['between'][0]['token_abs_logprob_error']['maximum'] == 0


@pytest.mark.parametrize('change', [
    'missing_run', 'duplicate_run', 'unknown_run', 'missing_response', 'duplicate_response',
    'unknown_response', 'truncated', 'appended_padding', 'token', 'record_hash',
    'positive', 'nan', 'inf', 'bool_token', 'bool_rep', 'bool_probability',
    'huge_int', 'duplicate_original', 'policy', 'instruction', 'finish_reason',
    'changed_original', 'original_bool_rep', 'original_bool_token', 'original_positive',
])
def test_rejects_incomplete_or_changed_evidence(change):
    records, runs, protocol = fixture()
    row = runs[0]['responses'][0]
    if change == 'missing_run': runs.pop()
    if change == 'duplicate_run': runs[1] = deepcopy(runs[0])
    if change == 'unknown_run': runs[0]['name'] = 'unknown'
    if change == 'missing_response': runs[0]['responses'].pop()
    if change == 'duplicate_response': runs[0]['responses'].append(deepcopy(row))
    if change == 'unknown_response': row['decision_id'] = 'other'
    if change == 'truncated':
        row['token_ids'].pop()
        row['log_probs'].pop()
    if change == 'appended_padding':
        row['token_ids'].append(0)
        row['log_probs'].append(0.)
    if change == 'token': row['token_ids'][0] = 123
    if change == 'record_hash': row['record_sha256'] = 'c'*64
    if change == 'positive': row['log_probs'][0] = 1e-8
    if change == 'nan': row['log_probs'][0] = float('nan')
    if change == 'inf': row['log_probs'][0] = -float('inf')
    if change == 'bool_token': row['token_ids'][0] = True
    if change == 'bool_rep': row['replicate'] = False
    if change == 'bool_probability': row['log_probs'][0] = False
    if change == 'huge_int': row['log_probs'][0] = -(10**400)
    if change == 'duplicate_original': records.append(deepcopy(records[0]))
    if change == 'policy': records[0]['policy_sha256'] = 'c'*64
    if change == 'instruction': records[0]['instruction_sha256'] = 'c'*64
    if change == 'finish_reason': records[0]['finish_reason'] = 'abort'
    if change == 'changed_original': records[0]['log_probs'][0] = -1.1
    if change == 'original_bool_rep': records[0]['replicate'] = False
    if change == 'original_bool_token': records[0]['token_ids'][0] = True
    if change == 'original_positive': records[0]['log_probs'][0] = 1e-8
    # Rebind the roster to test inner identity validation, not just its digest.
    protocol['records_sha256'] = digest(records)
    with pytest.raises(ValueError):
        audit((records, runs, protocol))


@pytest.mark.parametrize('key,value', [
    ('max_abs_logprob_error', -.1), ('max_abs_logprob_error', True),
    ('max_abs_logprob_error', float('inf')), ('max_between_run_logprob_error', -1),
    ('token_ratio_lower', 0), ('token_ratio_lower', 1.1),
    ('token_ratio_upper', .9), ('token_ratio_upper', float('inf')),
    ('run_names', ['micro1', 'micro1']), ('run_names', ['micro1']),
    ('schema', 'other'), ('policy_sha256', 'invalid'),
])
def test_no_implicit_or_invalid_tolerances(key, value):
    args = fixture()
    args[2][key] = value
    with pytest.raises(ValueError):
        audit(args)


def test_protocol_hash_and_records_hash_are_required():
    records, runs, protocol = fixture()
    with pytest.raises(ValueError):
        compare(records, runs, protocol=protocol, protocol_sha256='0'*64)
    records.reverse()
    with pytest.raises(ValueError):
        compare(records, runs, protocol=protocol, protocol_sha256=digest(protocol))
    del protocol['token_ratio_lower']
    with pytest.raises(ValueError):
        audit((records, runs, protocol))


def test_signed_cancellation_cannot_hide_token_ratio_failure():
    args = fixture()
    args[1][0]['responses'][0]['log_probs'] = [-.5, -2.5]
    args[2]['max_abs_logprob_error'] = 1.
    args[2]['max_between_run_logprob_error'] = 1.
    result = audit(args)
    row = result['runs'][0]['responses'][0]
    assert row['response_ratio']['value'] == 1
    assert row['token_ratio_bounds_violations'] == 2
    assert not row['passed'] and not result['numerical_gate_passed']


def test_pairwise_gate_can_fail_when_each_run_matches_old_tolerance():
    args = fixture()
    args[1][0]['responses'][0]['log_probs'][0] = -.875
    args[1][1]['responses'][0]['log_probs'][0] = -1.125
    result = audit(args)
    assert all(run['passed'] for run in result['runs'])
    assert not result['between'][0]['passed']
    assert result['between'][0]['token_abs_logprob_error']['maximum'] == .25
    assert not result['numerical_gate_passed']


def test_original_json_precision_is_not_replaced_by_float32():
    args = fixture([[-1.00000001]])
    args[1][0]['responses'][0]['log_probs'] = [-1.]
    args[2]['max_abs_logprob_error'] = 1e-9
    result = audit(args)
    assert not result['numerical_gate_passed']
    assert result['runs'][0]['token_abs_logprob_error']['maximum'] == pytest.approx(1e-8)


def test_token_weighting_and_response_weighting_remain_distinct():
    args = fixture()
    args[1][0]['responses'][1]['log_probs'][0] += .125
    result = audit(args)['runs'][0]
    assert result['token_abs_logprob_error']['mean'] == pytest.approx(.125/6)
    assert result['response_mean_abs_logprob_error'] == pytest.approx(.125/3)
    assert result['token_abs_logprob_error']['p99_nearest_rank'] == .125


@pytest.mark.parametrize('direction', [-1, 1])
def test_sequence_ratio_extremes_explicit_without_clipping(direction):
    args = fixture([[-1000.]*2])
    args[1][0]['responses'][0]['log_probs'] = [-1000.+direction*400]*2
    result = audit(args)
    ratio = result['runs'][0]['responses'][0]['response_ratio']
    assert ratio['overflow'] == (direction == 1)
    assert ratio['underflow'] == (direction == -1)
    assert ratio['value'] == (None if direction == 1 else 0.)
    json.dumps(result, allow_nan=False)


def test_nonrepresentable_sum_rejected_instead_of_silent_infinity():
    args = fixture([[-1e308]*2])
    args[1][0]['responses'][0]['log_probs'] = [0., 0.]
    with pytest.raises(ValueError, match='nonfinite log-ratio arithmetic'):
        audit(args)


def test_all_registered_pairs_and_exact_zero_tolerance():
    args = fixture()
    extra = deepcopy(args[1][0]); extra['name'] = 'micro4'
    args[1].append(extra); args[2]['run_names'].append('micro4')
    args[2].update(max_abs_logprob_error=0, max_between_run_logprob_error=0,
                   token_ratio_lower=1, token_ratio_upper=1)
    report = audit(args)
    assert report['numerical_gate_passed'] and len(report['between']) == 3


def test_tail_percentile_does_not_replace_maximum_gate():
    args = fixture([[-2.]*100])
    args[1][0]['responses'][0]['log_probs'][-1] = -1.
    result = audit(args)
    stats = result['runs'][0]['token_abs_logprob_error']
    assert stats['p99_nearest_rank'] == 0
    assert stats['maximum'] == 1 and stats['mean'] == .01
    assert not result['numerical_gate_passed']


@pytest.mark.parametrize('direction', [-1, 1])
def test_token_ratio_overflow_is_explicit_and_fails_gate(direction):
    args = fixture([[-1000.]])
    args[1][0]['responses'][0]['log_probs'] = [-1000.+direction*800]
    result = audit(args)
    ratio = result['runs'][0]['token_ratio_maximum' if direction == 1 else 'token_ratio_minimum']
    assert ratio['overflow'] == (direction == 1)
    assert ratio['underflow'] == (direction == -1)
    assert not result['numerical_gate_passed']
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('fail', [False, True])
def test_cli_saved_evidence_and_exit_status(tmp_path, fail):
    records, runs, protocol = fixture()
    if fail: runs[0]['responses'][0]['log_probs'][0] = -2.
    argv = []
    for name, value in [('records', records), ('runs', runs), ('protocol', protocol)]:
        path = tmp_path / (name+'.json'); path.write_text(json.dumps(value))
        argv.extend(['--'+name, str(path)])
    output = tmp_path / 'report.json'
    argv.extend(['--protocol-sha256', digest(protocol), '--output', str(output)])
    assert main(argv) == int(fail)
    saved = output.read_bytes()
    assert json.loads(saved)['numerical_gate_passed'] == (not fail)
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2 and output.read_bytes() == saved


@pytest.mark.parametrize('bad_json', ['{"x":1,"x":2}', '[NaN]'])
def test_cli_rejects_ambiguous_json_before_writing(tmp_path, bad_json):
    path = tmp_path / 'bad.json'; path.write_text(bad_json)
    out = tmp_path / 'result.json'
    argv = ['--records', str(path), '--runs', str(path), '--protocol', str(path),
            '--protocol-sha256', 'a'*64, '--output', str(out)]
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2 and not out.exists()

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from multimodal_jev.omni_probability_audit import compare
from multimodal_jev.omni_probability_batch import build_components, make_probability_batch
from multimodal_jev.omni_probability_capture import capture, recompute
from multimodal_jev.sfc_reward_batch import digest
from test_omni_actor_forward import Batch
from test_omni_probability_batch import fixture


def batch_fixture():
    args = fixture()
    parts = build_components(**args)
    return args, Batch(parts['tensors'], parts['non_tensors'], parts['meta_info'])


def protocol(records):
    return dict(schema='omni-probability-comparison-v1', records_sha256=digest(records),
                policy_sha256='a'*64, instruction_sha256='b'*64,
                run_names=['micro1', 'micro2'], max_abs_logprob_error=0,
                token_ratio_lower=1, token_ratio_upper=1, max_between_run_logprob_error=0)


def actor_fixture(*, token_specific=False):
    actor = SimpleNamespace(use_remove_padding=False, use_fused_kernels=False,
                            use_prefix_grouper=False, use_dynamic_bsz=False,
                            ulysses_sequence_parallel_size=1)
    calls = []

    def compute(inputs, calculate_entropy):
        assert not calculate_entropy and not torch.is_grad_enabled()
        assert 'old_log_probs' not in inputs.batch
        assert 'record_identity' not in inputs.non_tensor_batch
        assert inputs.meta_info['temperature'] == 1.
        calls.append(inputs.meta_info['micro_batch_size'])
        # Deterministic synthetic output, independent of old_log_probs.
        tokens = inputs.batch['responses']
        result = (-tokens.double()/8 if token_specific else
                  -torch.arange(1, tokens.shape[1]+1).double().expand_as(tokens).clone())
        result[tokens == 0] = float('nan')
        return {'log_probs': result}

    actor.compute_log_prob = compute
    return actor, calls


def test_forward_capture_compare_without_exposing_old_or_padding():
    args, batch = batch_fixture()
    actor, calls = actor_fixture()
    original = deepcopy(args['records'])
    old = batch.batch['old_log_probs'].clone(); meta = deepcopy(batch.meta_info)
    runs = [recompute(actor, batch, args['records'], name=f'micro{size}',
                      micro_batch_size=size, pad_token_id=0) for size in (1, 2)]
    p = protocol(original)
    report = compare(original, runs, protocol=p, protocol_sha256=digest(p))
    assert report['numerical_gate_passed'] and report['tokens'] == 10
    assert calls == [1, 2] and args['records'] == original
    assert torch.equal(batch.batch['old_log_probs'], old) and batch.meta_info == meta
    assert runs[0]['responses'][1]['token_ids'] == [20, 99]  # Preserve EOS.
    assert len(runs[0]['responses'][1]['log_probs']) == 2  # Exclude padding.


def test_actor_float64_precision_is_preserved_in_export():
    args, batch = batch_fixture()
    logp = batch.batch['old_log_probs'].double()
    logp[0, 0] = -1.000000001
    result = capture(batch, logp, args['records'], name='precision')
    assert result['responses'][0]['log_probs'][0] == -1.000000001


@pytest.mark.parametrize('kind', [
    'duplicate', 'unknown', 'bool_rep', 'identity_length', 'token', 'mask_hole',
    'mask_truncate', 'old', 'old_dtype', 'records_hash', 'policy', 'instruction',
    'diagnostic', 'synthetic', 'targets', 'boundary', 'shape', 'integer', 'nan', 'positive',
])
def test_export_rejects_misaligned_or_invalid_evidence(kind):
    args, batch = batch_fixture()
    logp = batch.batch['old_log_probs'].clone()
    identities = batch.non_tensor_batch['record_identity']
    if kind == 'duplicate': identities[1] = identities[0]
    if kind == 'unknown': identities[0] = ('unknown', 'calls', 1)
    if kind == 'bool_rep': identities[0] = ('one-audio', 'calls', True)
    if kind == 'identity_length': identities[0] = ('one-audio', 'calls')
    if kind == 'token': batch.batch['responses'][0, 0] = 9
    if kind == 'mask_hole': batch.batch['response_mask'][0] = torch.tensor([1, 0, 1])
    if kind == 'mask_truncate': batch.batch['response_mask'][0, -1] = 0
    if kind == 'old': batch.batch['old_log_probs'][0, 0] -= .5
    if kind == 'old_dtype': batch.batch['old_log_probs'] = batch.batch['old_log_probs'].double()
    if kind == 'records_hash': batch.meta_info['records_sha256'] = 'c'*64
    if kind in ('policy', 'instruction'): batch.meta_info[kind+'_sha256'] = 'c'*64
    if kind == 'diagnostic': batch.meta_info['diagnostic_only'] = False
    if kind == 'synthetic': batch.meta_info['synthetic_old_log_probs'] = True
    if kind == 'targets': batch.meta_info['targets_read'] = True
    if kind == 'boundary': batch.meta_info['waveform_boundary'] = 'raw'
    if kind == 'shape': logp = logp[:1]
    if kind == 'integer': logp = logp.long()
    if kind == 'nan': logp[0, 0] = float('nan')
    if kind == 'positive': logp[0, 0] = .1
    with pytest.raises(ValueError): capture(batch, logp, args['records'], name='micro1')


def test_invalid_identity_rejected_before_actor_call():
    args, batch = batch_fixture()
    actor, calls = actor_fixture()
    batch.non_tensor_batch['record_identity'][0] = ('unknown', 'calls', 1)
    with pytest.raises(ValueError):
        recompute(actor, batch, args['records'], name='micro1', micro_batch_size=1, pad_token_id=0)
    assert calls == []


def test_actor_failure_not_retried():
    args, batch = batch_fixture()
    actor, _ = actor_fixture(); calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError('deliberate synthetic forward failure')

    actor.compute_log_prob = fail
    with pytest.raises(RuntimeError, match='deliberate'):
        recompute(actor, batch, args['records'], name='micro1', micro_batch_size=1, pad_token_id=0)
    assert calls == [1]


def test_real_dataproto_reorder_forward_export_and_compare():
    pytest.importorskip('verl')
    args = fixture()
    for r in args['records']:
        r['log_probs'] = [-token/8 for token in r['token_ids']]
    batch = make_probability_batch(**args)
    actor, calls = actor_fixture(token_specific=True)
    first = recompute(actor, batch, args['records'], name='micro1', micro_batch_size=1, pad_token_id=0)
    reordered = batch[[2, 0, 3, 1]]
    second = recompute(actor, reordered, args['records'], name='micro2', micro_batch_size=2, pad_token_id=0)
    assert [r['replicate'] for r in second['responses']] == [0, 1, 1, 0]
    p = protocol(args['records'])
    report = compare(args['records'], [first, second], protocol=p, protocol_sha256=digest(p))
    assert report['numerical_gate_passed'] and calls == [1, 2]
    # Both rows have equal lengths, but different audio/replicate/token identity.
    # Reordering only their probability arrays must be caught numerically.
    wrong = deepcopy(second)
    a, b = wrong['responses'][0], wrong['responses'][3]
    a['log_probs'], b['log_probs'] = b['log_probs'], a['log_probs']
    failed = compare(args['records'], [first, wrong], protocol=p, protocol_sha256=digest(p))
    assert not failed['numerical_gate_passed']
    assert failed['runs'][1]['failed_responses'] == 2
    with pytest.raises(ValueError, match='roster'):
        recompute(actor, batch[[0, 1]], args['records'], name='partial', micro_batch_size=1, pad_token_id=0)
    assert calls == [1, 2]

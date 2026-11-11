from copy import deepcopy

import pytest
import torch

from multimodal_jev.omni_logprob_observer import PINNED_SOURCE, bind_observations
from test_omni_probability_capture import batch_fixture


def fixture(size=2):
    args, batch = batch_fixture()
    tokens = batch.batch['responses']
    native = -tokens.double() / 8 - .125
    report = dict(source_sha256=PINNED_SOURCE, flash_available=False,
                  npu_available=True, calls=[])
    for index, start in enumerate(range(0, len(tokens), size)):
        labels = tokens[start:start+size]
        values = native[start:start+size]
        report['calls'].append(dict(call=index, logits_shape=[*labels.shape, 128],
            positions=labels.numel(), labels_dtype='torch.int64',
            logits_device='npu:0', output_device='npu:0',
            reference_dtype='torch.float32', reference_device='cpu',
            label_ids=labels.tolist(), native_log_probs=values.tolist(),
            reference_log_probs=(values.float()-.01).tolist()))
    native = native.masked_fill(~batch.batch['response_mask'].bool(), 0)
    return args, batch, native, report


@pytest.mark.parametrize('size', [1, 2, 3, 4])
def test_binding_preserves_identity_masks_and_valid_native_values(size):
    args, batch, native, report = fixture(size)
    before = deepcopy(report)
    result = bind_observations(batch, native, args['records'], report,
                               name='native', micro_batch_size=size)
    assert report == before
    identities = [key for b in result['microbatch_bindings'] for key in b['record_identities']]
    assert identities == [list(key) for key in batch.non_tensor_batch['record_identity']]
    for i, (n, r) in enumerate(zip(result['native']['responses'], result['reference']['responses'])):
        length = len(args['records'][i]['token_ids'])
        assert n['log_probs'] == native[i, :length].tolist()
        assert len(r['log_probs']) == length
        assert n['record_sha256'] == r['record_sha256']
        assert all(a != b for a, b in zip(n['log_probs'], r['log_probs']))


@pytest.mark.parametrize('kind', ['missing', 'extra', 'order', 'labels', 'bool_label',
    'shape', 'native', 'reference_nan', 'device', 'source', 'mask'])
def test_rejects_incomplete_or_mismatched_call_evidence(kind):
    args, batch, native, report = fixture()
    c = report['calls'][0]
    if kind == 'missing': report['calls'].pop()
    if kind == 'extra': report['calls'].append(deepcopy(c))
    if kind == 'order': report['calls'].reverse()
    if kind == 'labels': c['label_ids'][0][0] += 1
    if kind == 'bool_label': c['label_ids'][0][0] = True
    if kind == 'shape': c['logits_shape'][0] += 1
    if kind == 'native': c['native_log_probs'][0][0] -= .1
    if kind == 'reference_nan': c['reference_log_probs'][0][0] = float('nan')
    if kind == 'device': c['logits_device'] = 'cpu'
    if kind == 'source': report['source_sha256'] = 'wrong'
    if kind == 'mask': batch.batch['response_mask'][0][0] = 0
    with pytest.raises(ValueError):
        bind_observations(batch, native, args['records'], report,
                          name='native', micro_batch_size=2)


def test_scoped_recomputation_calls_actor_once_and_binds_after_restore(monkeypatch):
    from contextlib import contextmanager
    from multimodal_jev import omni_logprob_observer, omni_actor_forward
    from multimodal_jev.omni_probability_capture import recompute_native_observed
    args, batch, native, report = fixture()
    events = []

    @contextmanager
    def observe():
        events.append('enter')
        try:
            yield report
        finally:
            events.append('restore')

    def forward(actor, data, **kwargs):
        assert not torch.is_grad_enabled() and data is batch
        events.append('forward')
        return native

    original_bind = omni_logprob_observer.bind_observations
    def bind(*args, **kwargs):
        assert events == ['enter', 'forward', 'restore']
        return original_bind(*args, **kwargs)

    monkeypatch.setattr(omni_logprob_observer, 'observe_native_logprobs', observe)
    monkeypatch.setattr(omni_logprob_observer, 'bind_observations', bind)
    monkeypatch.setattr(omni_actor_forward, 'compute_log_probs', forward)
    result = recompute_native_observed(object(), batch, args['records'],
        name='native', micro_batch_size=2, pad_token_id=0)
    assert len(result['native']['responses']) == len(args['records'])
    assert events == ['enter', 'forward', 'restore']

import pytest
import torch

from multimodal_jev.context_decision_head import ContextDecisionHead
from multimodal_jev.context_pooling import ContextHistoryReadout


def case():
    torch.manual_seed(317)
    head = ContextDecisionHead(12, 3, 24).double().eval()
    readout = ContextHistoryReadout(head, 99, 'a'*64)
    hidden = torch.randn(14, 12, dtype=torch.float64)
    prompts = [torch.tensor([1, 2, 99, 3, 99]),
               torch.tensor([1, 4, 5, 99, 6, 7, 99, 8, 99])]
    extras = [dict(policy_id='a'*64, marker_positions=pos, expected_input_ids=p.tolist())
              for p, pos in zip(prompts, [[2, 4], [3, 6, 8]])]
    return readout, hidden, [5, 9], prompts, extras


def test_packed_requests_match_full_context_reference_and_do_not_mix_history():
    r, h, lengths, prompts, extras = case()
    outputs = r(h, lengths, prompts, extras)
    for i, chunk in enumerate(h.split(lengths)):
        expected = r.head(chunk[None], torch.tensor([extras[i]['marker_positions']]),
                          torch.ones(1, len(chunk), dtype=torch.bool))[0, :, None]
        torch.testing.assert_close(outputs[i], expected, rtol=0, atol=0)
    changed = h.clone()
    changed[:5] *= 10
    torch.testing.assert_close(r(changed, lengths, prompts, extras)[1], outputs[1], rtol=0, atol=0)
    changed = h.clone()
    # Change history direction: uniform scaling is removed by LayerNorm.
    changed[5:8] = torch.randn_like(changed[5:8])
    assert (r(changed, lengths, prompts, extras)[1] - outputs[1]).abs().max() > 1e-8
    # Reordering whole requests preserves outputs and their identity.
    reversed_out = r(torch.cat([h[5:], h[:5]]), lengths[::-1], prompts[::-1], extras[::-1])
    for a, b in zip(reversed_out, outputs[::-1]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize('fault', ['policy', 'prompt', 'missing', 'extra_marker', 'order',
                                  'partial', 'nan', 'boolean', 'extra_field'])
def test_bad_requests_fail_before_any_head_call(fault):
    r, h, lengths, prompts, extras = case()
    calls = []
    r.head.register_forward_pre_hook(lambda *args: calls.append(1))
    if fault == 'policy': extras[1]['policy_id'] = 'b'*64
    if fault == 'prompt': extras[1]['expected_input_ids'][0] = 7
    if fault == 'missing': extras[1]['marker_positions'] = [3, 8]
    if fault == 'extra_marker': extras[1]['marker_positions'] = [3, 5, 6, 8]
    if fault == 'order': extras[1]['marker_positions'] = [8, 6, 3]
    if fault == 'partial': lengths[1] = 8
    if fault == 'nan': h[-1, 0] = float('nan')
    if fault == 'boolean': extras[1]['expected_input_ids'][0] = True
    if fault == 'extra_field': extras[1]['gold'] = 1
    with pytest.raises(ValueError): r(h, lengths, prompts, extras)
    assert not calls


def test_cpu_bf16_probabilities_are_fp32_and_finite():
    r, h, lengths, prompts, extras = case()
    r.bfloat16()
    for output in r(h.bfloat16(), lengths, prompts, extras):
        assert output.dtype == torch.float32 and torch.isfinite(output).all()
        assert ((output >= 0) & (output <= 1)).all()

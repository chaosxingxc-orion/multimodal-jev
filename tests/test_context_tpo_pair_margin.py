import pytest
import torch

from multimodal_jev.context_tpo_pair_margin import compile_pairs, loss, verify_update
from multimodal_jev.context_tpo_rewards import projection
from tools.reports.build_tpo_actor_pairs import admit


def test_frozen_switch_uses_strict_reference_and_any_projected_rival():
    reference = torch.tensor([[.2, .7, .7], [.5, .5, .2], [.2, .7, .1], [.2, .7, .1]])
    projected = torch.tensor([[[.25, -.1, .2], [0., 1., 0.], [0., 1., 0.], [.25, 0., 0.]],
                              [[.5, 0., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]]])
    rivals, enabled = compile_pairs(reference, projected, torch.zeros(4, dtype=torch.long),
                                    torch.tensor([True, True, False, True]), .25)
    assert rivals.tolist() == [1, 1, 1, 1]
    assert enabled.tolist() == [[True, False, False, False], [False, False, False, True]]


def test_actor_gradient_and_padding_with_zero_weight_equivalence():
    x = torch.tensor([[0., .5, 0.], [0., .5, 0.], [.25, 0., 0.], [0., .5, 0.]], requires_grad=True)
    q = torch.tensor([[0., 1., 0.], [.5, 0., 0.], [0., 1., 0.], [0., 1., 0.]])
    gold, rival = torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.long)
    real = torch.tensor([True, True, True, False])
    baseline = (projection(x, q) * real).sum() / real.sum()
    gradients = torch.autograd.grad(baseline, x, retain_graph=True)[0]
    zero, counts = loss(x, q, gold, rival, torch.ones_like(real), real, dict(logit_margin=.25, weight=0.))
    zero_loss = baseline + zero.sum() / real.sum()
    assert torch.equal(baseline, zero_loss)
    assert torch.equal(gradients, torch.autograd.grad(zero_loss, x, retain_graph=True)[0])
    assert counts['active'].item() == 0
    added, counts = loss(x, q, gold, rival, torch.ones_like(real), real, dict(logit_margin=.25, weight=1/3))
    delta = torch.autograd.grad(baseline + added.sum() / real.sum(), x)[0] - gradients
    assert torch.allclose(delta, torch.tensor([[-1., 1., 0.], [-1., 1., 0.], [0., 0., 0.], [0., 0., 0.]]) / 9)
    assert dict((k, n.item()) for k, n in counts.items()) == dict(
        eligible=3, active=2, turned_positive=1, strengthened=1, still_nonpositive=0)


def test_tail_batch_receipt_requires_correct_denominator_and_counts():
    row = dict(step=2, epoch=0, real=3, metrics={
        'jev/reward_projection_kl': .125, 'jev/pair_margin': .0625, 'jev/tpo_loss': .1875},
        pair_margin=dict(eligible=2, active=2, turned_positive=0, strengthened=1, still_nonpositive=1))
    verify_update(row, [384, 3], [0, 0], [10, 2])
    row["pair_margin"]["active"] = 3
    with pytest.raises(ValueError, match="pair margin update"):
        verify_update(row, [384, 3], [0, 0], [10, 2])


def test_review_admission_drops_disagreement_and_requires_new_complete_pair():
    bindings = dict(
        pairs=[dict(pair_id='one', sides=[['l', 'l2'], ['r']])],
        hard=['hard'], replay=[], review_ids={uid: uid for uid in ('l', 'l2', 'r', 'hard')})
    eligible = {uid: dict(intent='a' if uid != 'r' else 'b') for uid in bindings['review_ids']}
    first = {uid: dict(status='unique', label=row['intent']) for uid, row in eligible.items()}
    second = dict(first, l2=dict(status='ambiguous', label=None))
    relation = {'one': dict(status='confirmed', evidence='One condition changed.')}
    selected, approved, pairs, rejected = admit(bindings, (first, second, relation), eligible, ['a', 'b'], {'r'})
    assert set(selected) == approved == {'l', 'r', 'hard'}
    assert selected['hard']['kind'] == 'hard_single'
    assert pairs[0]['sides'] == [['l'], ['r']]
    assert rejected == {'missing_or_nonunique_recording': 1}
    with pytest.raises(ValueError, match='absent from 206'):
        admit(bindings, (first, second, relation), eligible, ['a', 'b'], {'l2'})

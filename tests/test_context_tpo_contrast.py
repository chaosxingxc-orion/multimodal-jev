import torch

from multimodal_jev.context_tpo_contrast import ContrastSpec, contrast_utilities, project
from multimodal_jev.context_tpo_rewards import projection


def test_utilities_match_differentiating_explicit_contests():
    p = torch.tensor([[.14, .62, .13], [.34, .21, .12], [.37, .08, .61],
                      [.97, .42, .41]], dtype=torch.float64, requires_grad=True)
    gold = torch.tensor([0, 0, 1, 1])
    temp = .8
    score = p.log() / temp
    row_loss = sum(score[i, k] - score[i].logsumexp(0) for i, k in enumerate(gold))
    column_loss = sum(score[i, k] - torch.cat((score[i, k:k+1], score[gold != k, k])).logsumexp(0)
                      for i, k in enumerate(gold))
    row, = torch.autograd.grad(row_loss, p, retain_graph=True)
    col, = torch.autograd.grad(column_loss, p)
    result, skipped = contrast_utilities(p, gold, temp)
    assert skipped == 0
    torch.testing.assert_close(result["audio_to_label"], row)
    torch.testing.assert_close(result["label_to_audio"], col)
    assert not result["label_to_audio"][:, 2].count_nonzero()


def test_same_label_positives_do_not_compete_and_projection_is_bernoulli():
    p = torch.tensor([[0., .88, .22], [.99, .11, 1.], [.12, .31, .09]], dtype=torch.float64)
    gold = torch.tensor([0, 0, 0])
    spec = ContrastSpec(1., 1e-6, 1., .5, .5, 1.)
    q, audit = project(p, gold, spec)
    assert audit["column_positive_skipped"] == len(p)
    assert audit["boundary_clipped"] == 2
    idx = torch.tensor([2, 0, 1])
    shuffled, _ = project(p[idx], gold[idx], spec)
    torch.testing.assert_close(q[idx], shuffled)
    z = p.clamp(1e-6, 1 - 1e-6).logit().float().requires_grad_()
    loss = projection(z, q)
    assert (loss > 0).all()
    loss.sum().backward()
    assert (z.grad[:, 0] <= 0).all()
    assert (z.grad[:, 1:] >= 0).all()

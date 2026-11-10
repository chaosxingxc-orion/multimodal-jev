#!/usr/bin/env python3
"""CPU algebra/gradient check for a proposed class-weighted clipped objective.

Synthetic log probabilities only: no actor, optimizer or training dispatch.
Class weights come exclusively from the pinned complete learning audit.
"""
import argparse
import inspect
import json
from pathlib import Path

from multimodal_jev.baseline import atomic_json, file_hash
from probe_sfc_reward_balance import run as check_reward_balance


def run(audit_path, audit_hash):
    audit = check_reward_balance(audit_path, audit_hash)
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, compute_policy_loss_vanilla

    core_hash = file_hash(inspect.getfile(compute_policy_loss_vanilla))
    if core_hash != 'fadfafa83d94ba40d7ac2abeedb6e99ea0716b42f8744c421ecbb371db8b4d24':
        raise ValueError('pinned verl policy loss required')
    torch.set_num_threads(2)
    n, width = 16, 5
    lengths = torch.tensor([1, 2, 4, 5, 5, 3, 1, 2, 2, 5, 4, 1, 3, 1, 5, 2])
    mask = (torch.arange(width)[None, :] < lengths[:, None]).float()
    rewards = torch.tensor([1., 0., -1., 1., 1., 0., -1., 1., 1.2, 1., -.1, -1., 0., 0., 0., 0.])
    token_rewards = torch.zeros_like(mask)
    token_rewards[torch.arange(n), lengths - 1] = rewards
    advantage, _ = compute_grpo_outcome_advantage(
        token_rewards, mask, np.repeat(np.arange(4), 4), norm_adv_by_std_in_grpo=True)
    weights = torch.tensor([audit['class_weights'][key] for key in
                            ['active', 'inactive', 'active', 'inactive']]).repeat_interleave(4)
    index = torch.arange(n * width).reshape(n, width).float()
    features = torch.stack([torch.sin(index * .37), torch.cos(index * .23)], dim=-1)
    # Both signs and ratio regions; the -2 offset is a synthetic old logprob.
    delta = torch.tensor([0., .5, 1.5, -.6]).repeat(4)[:, None].expand(n, width)
    old = torch.full_like(mask, -2.)
    reports = {}

    def reference(theta, mode):
        ratio = torch.exp(delta + (features * theta).sum(-1))
        clipped = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(.8, 1.2))
        dual = torch.where(advantage < 0, torch.minimum(-3. * advantage, clipped), clipped)
        weighted = dual * weights[:, None] * mask
        if mode == 'token-mean':
            return weighted.sum() / mask.sum()
        return (weighted.sum(-1) / lengths).sum() / n

    def actual(theta, ids, mode, global_info, weighted=True):
        cfg = OmegaConf.create(dict(clip_ratio=.2, clip_ratio_low=None, clip_ratio_high=None,
                                    clip_ratio_c=3., global_batch_info=global_info))
        new = old[ids] + delta[ids] + (features[ids] * theta).sum(-1)
        adv = advantage[ids] * (weights[ids, None] if weighted else 1.)
        return compute_policy_loss_vanilla(old[ids], new, adv, mask[ids],
                                          loss_agg_mode=mode, config=cfg)[0]

    for mode in ('token-mean', 'seq-mean-token-mean'):
        theta = torch.tensor([.03, -.02], requires_grad=True)
        expected_loss = reference(theta, mode)
        expected_grad, = torch.autograd.grad(expected_loss, theta)
        full = actual(theta, list(range(n)), mode, {})
        full_grad, = torch.autograd.grad(full, theta)
        torch.testing.assert_close(full, expected_loss, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(full_grad, expected_grad, atol=2e-6, rtol=2e-5)
        # Deliberately unequal microbatch sizes and token counts. Sum losses;
        # do not divide again by the number of microbatches.
        chunks = [list(range(3)), list(range(3, 10)), list(range(10, n))]
        info = ({'dp_size': 1, 'batch_num_tokens': int(mask.sum())} if mode == 'token-mean'
                else {'dp_size': 1, 'global_batch_size': n})
        accumulated = sum(actual(theta, ids, mode, info) for ids in chunks)
        accumulated_grad, = torch.autograd.grad(accumulated, theta)
        torch.testing.assert_close(accumulated_grad, expected_grad, atol=2e-6, rtol=2e-5)
        wrong_local = sum(actual(theta, ids, mode, {}) for ids in chunks) / len(chunks)
        wrong_grad, = torch.autograd.grad(wrong_local, theta)
        unweighted = actual(theta, list(range(n)), mode, {}, weighted=False)
        unweighted_grad, = torch.autograd.grad(unweighted, theta)
        local_error = float((wrong_grad - expected_grad).abs().max())
        class_effect = float((unweighted_grad - expected_grad).abs().max())
        if min(local_error, class_effect) <= 1e-5:
            raise ValueError('negative controls must distinguish weighting and normalization')
        reports[mode] = dict(loss=float(full.detach()), reference_gradient=expected_grad.tolist(),
            actual_gradient=full_grad.tolist(), accumulated_gradient=accumulated_grad.tolist(),
            gradient_error=float((full_grad-expected_grad).abs().max()),
            microbatch_error=float((accumulated_grad-expected_grad).abs().max()),
            incorrect_local_mean_error=local_error, class_weight_gradient_effect=class_effect)
    difference = max(abs(a-b) for a,b in zip(reports['token-mean']['actual_gradient'],
                                             reports['seq-mean-token-mean']['actual_gradient']))
    if difference <= 1e-5:
        raise ValueError('response-length normalization control must be discriminating')
    return dict(schema='sfc-weighted-policy-loss-diagnostic-v1',audit_sha256=audit_hash,
        core_sha256=core_hash,class_weights=audit['class_weights'],reports=reports,
        token_vs_response_gradient_difference=difference,
        response_count=n,valid_tokens=int(mask.sum()),model_loaded=False,weight_updates=0,
        scope='Synthetic differentiable logprob fixture through actual verl clipped policy loss. '
              'Positive post-standardization weights scale the clipped policy term; global denominators '
              'preserve microbatch gradients. No model gradients, optimizer, DDP, efficacy or training.')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--audit-hash',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve prior result')
    result=run(a.audit,a.audit_hash);atomic_json(a.output,result)
    print(json.dumps(result,indent=2))

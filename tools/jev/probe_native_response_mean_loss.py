#!/usr/bin/env python3
"""Synthetic gradient audit of registered response loss, no model/optimizer."""
import argparse
import inspect
from pathlib import Path

from multimodal_jev.baseline import atomic_json, file_hash


def run(output):
    import numpy as np
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, compute_policy_loss_vanilla, POLICY_LOSS_REGISTRY
    from multimodal_jev.verl_native_loss import register_native_response_mean_loss
    torch.set_num_threads(2)
    core_hash = file_hash(inspect.getfile(compute_policy_loss_vanilla))
    if core_hash != 'fadfafa83d94ba40d7ac2abeedb6e99ea0716b42f8744c421ecbb371db8b4d24':
        raise ValueError('pinned real verl loss required')
    n, width = 12, 7
    lengths = torch.tensor([1, 2, 7, 3, 1, 6, 5, 2, 7, 1, 4, 6])
    mask = (torch.arange(width)[None, :] < lengths[:, None]).float()
    rewards = torch.tensor([1., -1., 0., 1., 0., -1., 0., 0., 0., -1., 1., .5])
    token_rewards = torch.zeros_like(mask); token_rewards[torch.arange(n), lengths-1] = rewards
    adv, _ = compute_grpo_outcome_advantage(token_rewards, mask,
        np.repeat(np.arange(4), 3), norm_adv_by_std_in_grpo=True)
    # Deliberately synthetic positive class weights; no empirical policy claim.
    weights = torch.tensor([2., .5, 1., 1.5]).repeat_interleave(3)
    weighted_adv = adv * weights[:, None]
    index = torch.arange(n*width).reshape(n, width).float()
    features = torch.stack([torch.sin(index*.37), torch.cos(index*.23)], -1)
    delta = torch.tensor([0., .5, 1.5, -.6]).repeat(3)[:, None].expand(n, width)
    old = torch.full_like(mask, -4.)
    theta = torch.tensor([.03, -.02], requires_grad=True)
    new = old + delta + (features*theta).sum(-1)
    ratio = (new-old).exp()
    clipped = torch.maximum(-weighted_adv*ratio, -weighted_adv*ratio.clamp(.8, 1.2))
    dual = torch.where(weighted_adv < 0, torch.minimum(-3.*weighted_adv, clipped), clipped)
    expected = ((dual*mask).sum(-1)/lengths).mean()
    expected_grad, = torch.autograd.grad(expected, theta, retain_graph=True)

    def config(micro):
        return OmegaConf.create(dict(use_dynamic_bsz=False, ppo_epochs=1, use_kl_loss=False,
            entropy_coeff=0., calculate_entropy=False, use_rollout_log_probs=True,
            global_batch_info={}, ppo_mini_batch_size=n, ppo_micro_batch_size_per_gpu=micro,
            jev_batch_num_tokens=int(mask.sum()), jev_batch_num_responses=n,
            clip_ratio=.2, clip_ratio_low=None, clip_ratio_high=None, clip_ratio_c=3.))

    dist.init_process_group('gloo', init_method=(output/'gloo-init').resolve().as_uri(), rank=0, world_size=1)
    reports = {}
    try:
        name = register_native_response_mean_loss(); function = POLICY_LOSS_REGISTRY[name]
        for micro in (1, 2, 3, 4, 6, 12):
            cfg = config(micro); before = OmegaConf.to_container(cfg)
            # Reproduce the native static loop's loss / accumulation_steps.
            terms = [function(old[s:s+micro], new[s:s+micro], weighted_adv[s:s+micro],
                     mask[s:s+micro], config=cfg)[0]/(n//micro) for s in range(0, n, micro)]
            loss = sum(terms); grad, = torch.autograd.grad(loss, theta, retain_graph=True)
            torch.testing.assert_close(loss, expected, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(grad, expected_grad, atol=2e-6, rtol=2e-5)
            if OmegaConf.to_container(cfg) != before:
                raise ValueError('adapter mutated config')
            reports[str(micro)] = dict(loss=float(loss.detach()), gradient=grad.tolist(),
                max_gradient_error=float((grad-expected_grad).abs().max()))
        token_loss = (dual*mask).sum()/mask.sum()
        unweighted = compute_policy_loss_vanilla(old,new,adv,mask,
            loss_agg_mode='seq-mean-token-mean',config=config(n))[0]
        token_grad, = torch.autograd.grad(token_loss, theta, retain_graph=True)
        unweighted_grad, = torch.autograd.grad(unweighted, theta)
        controls = dict(token_mean_gradient_difference=float((token_grad-expected_grad).abs().max()),
                        unweighted_gradient_difference=float((unweighted_grad-expected_grad).abs().max()))
        if min(controls.values()) <= 1e-5:
            raise ValueError('negative controls do not distinguish proposed objective')
        return dict(status='passed', cases=reports, controls=controls,
            reference_loss=float(expected.detach()), reference_gradient=expected_grad.tolist(),
            core_sha256=core_hash, adapter_sha256=file_hash(inspect.getfile(function)),
            synthetic_class_weights=weights.tolist(), model_loaded=False, weight_updates=0,
            scope='Real registered verl loss and synthetic gradients across six static microbatch sizes; native accumulation divisor reproduced, no actual actor loop, optimizer, model training or distributed acceptance.')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); a=p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    try:
        atomic_json(a.output/'result.json', run(a.output))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json', dict(exit_code=1,error=repr(exc))); raise
    atomic_json(a.output/'terminal.json', dict(exit_code=0))

"""Independent saved-group audit for both frozen RL algorithms.

Works on one complete global batch before DP splitting. No model calls or
changes to rewards/advantages. Padded slots are checked but excluded from
reported learning statistics; zero-signal real queries remain in denominators.
"""
import torch

from tools.jev.context_rollout_audit import verify_sample


def expected_advantages(rewards, algorithm):
    if algorithm not in ('grpo_v1', 'rloo_v1'):
        raise ValueError('registered frozen algorithm required')
    if (rewards.shape != (1, 16) or rewards.dtype != torch.float32
            or not torch.isfinite(rewards).all()):
        raise ValueError('one complete finite FP32 G16 reward group required')
    if algorithm == 'grpo_v1':
        return (rewards-rewards.mean(-1, keepdim=True))/(rewards.std(-1, keepdim=True)+1e-6)
    return torch.stack([rewards[:, j] - torch.cat(
        (rewards[:, :j], rewards[:, j+1:]), -1).mean(-1) for j in range(16)], -1)


def audit(groups, receipts, targets, *, algorithm, update, policy_id, reference_policy_id):
    if algorithm not in ('grpo_v1', 'rloo_v1'):
        raise ValueError('registered frozen algorithm required')
    ids = [g['request_id'] for g in groups]
    rid = [r['request_id'] for r in receipts]
    if (not groups or len(set(ids)) != len(ids) or len(set(rid)) != len(rid)
            or set(ids) != set(rid) or set(ids) != set(targets)):
        raise ValueError('complete unique groups, receipts and targets required')
    entries = {r['request_id']:r for r in receipts}
    real = []
    for g in groups:
        target = targets[g['request_id']]
        labels, observed, actions = g['labels'], g['observed'], g['actions']
        if (labels.dtype != torch.bool or labels.shape != (1, 60)
                or observed.dtype != torch.bool or observed.shape != labels.shape or not observed.all()
                or labels.sum().item() != 1 or labels.tolist() != [target['labels']]
                or type(g['real_request']) is not bool or g['real_request'] is not target['real_request']
                or g['request_weight'] != target['weight']
                or g['request_weight'] != (1. if g['real_request'] else 0.)
                or actions.shape != (1, 16, 60) or actions.dtype != torch.int64
                or not ((actions == 0) | (actions == 1)).all()):
            raise ValueError('complete exclusive-event labels, binary actions and frozen weights required')
        selected = actions.bool()
        rewards = 2 * (selected & labels[:, None, :]).sum(-1).float() / (
            selected.sum(-1) + labels[:, None, :].sum(-1))
        if (g['rewards'].dtype != torch.float32 or g['rewards'].shape != rewards.shape
                or not torch.equal(rewards, g['rewards'])):
            raise ValueError('recorded set rewards differ')
        expected = expected_advantages(rewards, algorithm)
        if (g['advantages'].shape != expected.shape or g['advantages'].dtype != torch.float32
                or not torch.allclose(g['advantages'], expected, atol=2e-6, rtol=0)):
            raise ValueError('recorded algorithm advantages differ')
        verify_sample(entries[g['request_id']], g, update=update, policy_id=policy_id,
                      reference_policy_id=reference_policy_id)
        logjoint = torch.zeros(1, 16)
        for j, logp in enumerate(g['behavior']):
            logjoint += logp.gather(1, actions[:, :, j])
        if (g['old_joint'].shape != logjoint.shape or g['old_joint'].dtype != torch.float32
                or not torch.allclose(logjoint, g['old_joint'], atol=2e-5, rtol=0)):
            raise ValueError('complete joint behavior likelihood differs')
        if g['real_request']:
            real.append(dict(reward_mean=rewards.mean().item(), reward_std=rewards.std().item(),
                             zero_signal=bool((rewards == rewards[:, :1]).all()),
                             advantage_abs_mean=expected.abs().mean().item()))
    if not real:
        raise ValueError('nonempty real learning population required')
    return dict(algorithm=algorithm, update=update, checked_slots=len(groups), real_queries=len(real),
                padding_slots=len(groups)-len(real),
                mean_reward=sum(r['reward_mean'] for r in real)/len(real),
                mean_group_reward_std=sum(r['reward_std'] for r in real)/len(real),
                zero_signal_groups=sum(r['zero_signal'] for r in real),
                zero_signal_fraction=sum(r['zero_signal'] for r in real)/len(real),
                mean_abs_advantage=sum(r['advantage_abs_mean'] for r in real)/len(real),
                scope='Saved group/reward/estimator/sample audit, not model execution or task efficacy')

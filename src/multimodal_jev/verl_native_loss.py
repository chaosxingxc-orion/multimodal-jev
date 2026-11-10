"""Explicit token weighting for the pinned native actor's static microbatch loop.

Single-rank, one-minibatch/one-epoch policy loss only. No distributed claim and
no changes to upstream sources. The full-group advantages must be computed
before slicing. Native update_policy divides each loss by accumulation steps;
this registered loss compensates exactly once and uses the full token count.
"""
import copy

LOSS_NAME = 'jev_static_token_mean_v1'


def native_token_mean_loss(old_log_prob, log_prob, advantages, response_mask,
                           loss_agg_mode='token-mean', config=None,
                           rollout_is_weights=None):
    import torch
    import torch.distributed as dist
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

    if config is None:
        raise ValueError('explicit actor config required')
    if not dist.is_initialized() or dist.get_world_size() != 1:
        raise ValueError('single initialized rank required; distributed weighting not validated')
    if (config.use_dynamic_bsz or config.ppo_epochs != 1 or config.use_kl_loss
            or config.entropy_coeff != 0 or config.calculate_entropy
            or not config.use_rollout_log_probs or loss_agg_mode != 'token-mean'
            or config.global_batch_info or rollout_is_weights is not None):
        raise ValueError('unsupported native loss contract')
    size = config.ppo_mini_batch_size
    micro = config.ppo_micro_batch_size_per_gpu
    tokens = config.jev_batch_num_tokens
    rows = config.jev_batch_num_responses
    if (any(type(v) is not int or v <= 0 for v in (size, micro, tokens, rows))
            or rows != size or micro > size or size % micro != 0):
        raise ValueError('one complete divisible static minibatch required')
    if (response_mask.ndim != 2 or response_mask.shape[0] != micro
            or not torch.all((response_mask == 0) | (response_mask == 1))
            or not torch.all(response_mask.sum(-1) > 0)):
        raise ValueError('complete nonempty binary response masks required')
    local_tokens = int(response_mask.sum())
    if local_tokens > tokens:
        raise ValueError('local tokens exceed declared batch tokens')
    # Keep the actual global count intact. Compensate the native loop's
    # 1/gradient_accumulation after evaluating vanilla globally-normalized loss.
    vanilla = copy.deepcopy(config)
    vanilla.global_batch_info = {'dp_size': 1, 'batch_num_tokens': tokens}
    loss, metrics = compute_policy_loss_vanilla(
        old_log_prob, log_prob, advantages, response_mask,
        loss_agg_mode=loss_agg_mode, config=vanilla,
    )
    return loss * (size // micro), metrics


def register_native_token_mean_loss():
    """Register once, rejecting a collision instead of replacing another loss."""
    from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY, register_policy_loss

    existing = POLICY_LOSS_REGISTRY.get(LOSS_NAME)
    if existing is not None and existing is not native_token_mean_loss:
        raise ValueError('native loss registration collision')
    if existing is None:
        register_policy_loss(LOSS_NAME)(native_token_mean_loss)
    return LOSS_NAME

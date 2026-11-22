"""A veRL policy-loss extension for the exact full-action projection."""
import numpy as np
import torch
import torch.distributed as dist

from .context_native_actor import _active
from .context_tpo import Objective, objectives

MODE = 'jev_full_slate_tpo_v1'


def pack(items):
    from verl import DataProto
    n = len(items)
    # Required by the pinned veRL DataParallelPPOActor transport. The TPO loss
    # reads full-slate logits from the audio forward and ignores PPO columns.
    data = np.empty(n, dtype=object)
    data[:] = items
    zeros = torch.zeros(n, 1)
    token = torch.zeros(n, 1, dtype=torch.long)
    return DataProto.from_dict(tensors=dict(responses=token, response_mask=torch.ones_like(token),
        old_log_probs=zeros, advantages=zeros, input_ids=token, attention_mask=torch.ones_like(token),
        position_ids=token), non_tensors={'multi_modal_inputs':data},
        meta_info=dict(temperature=1.,use_dynamic_bsz=False,pad_token_id=0))


def admit(batch, config):
    items = batch.non_tensor_batch['multi_modal_inputs']
    if len(items) != config.ppo_mini_batch_size:
        raise ValueError('incomplete local TPO batch')
    joint = config.get('jev_joint_tpo')
    if joint:
        micro = config.ppo_micro_batch_size_per_gpu
        if len(items) % (2*micro):
            raise ValueError('complete pairs of view microbatches required')
        pairs = []
        for start in range(0, len(items), 2*micro):
            for n, c in zip(items[start:start+micro], items[start+micro:start+2*micro], strict=True):
                if ((n['choice_mode'], c['choice_mode']) != ('noul','open') or any(n[k]!=c[k] for k in
                      ('request_id','real_request','request_weight','global_real','policy_id'))
                      or n['request_weight'] != float(n['real_request'])):
                    raise ValueError('paired views differ in recording identity or weight')
                pairs.append(n)
        items = pairs
    groups = [None] * dist.get_world_size()
    dist.all_gather_object(groups, [(x['request_id'], x['real_request'], x['global_real']) for x in items])
    rows = [row for group in groups for row in group]
    count = sum(real for _, real, _ in rows)
    if (count < 1 or any(den != count for _, _, den in rows) or
            len(set(uid for uid, _, _ in rows)) != len(rows)):
        raise ValueError('global population or TPO denominator differs')
    return dict(global_real_requests=count)


def registered_loss(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode='token-mean',
                    config=None, rollout_is_weights=None, **kwargs):
    state = _active.get()
    if (state is None or state['consumed'] or state['log_probs'] is not log_prob
            or config.policy_loss.loss_mode != MODE or rollout_is_weights is not None):
        raise ValueError('loss must consume exactly one audio forward')
    state['consumed'] = True
    items, logits = state['payload'], state['logits']
    device = logits.device
    if 'macro_prior' in items[0]:
        with torch.no_grad():
            prior=torch.stack([x['macro_prior'] for x in items]).to(device)
            if (any(x['policy_id']!=config.jev_policy_id for x in items) or prior.shape!=logits.shape
                    or not torch.allclose(prior.sigmoid(),logits.detach().float().sigmoid(),rtol=0.,
                                          atol=config.jev_context_parity.probability_max_abs)):
                raise ValueError('training forward differs from current macro reference')
    flags = [x['paired'] for x in items]
    confusion = [x['confusion'] for x in items]
    partners = [(i ^ 1) if f or confusion[i] else i for i, f in enumerate(flags)]
    for i, j in enumerate(partners):
        if confusion[i] and (flags[i] or not confusion[j] or items[i]['gold']==items[j]['gold'] or
                not (items[i]['real_request'] and items[j]['real_request']) or
                items[i]['audio_sha256']==items[j]['audio_sha256']):
            raise ValueError('invalid contrast partner')
        if flags[i] and (partners[j] != i or items[i]['group'] is None or
                items[i]['group'] != items[j]['group'] or items[i]['gold'] != items[j]['gold'] or
                items[i]['audio_sha256'] == items[j]['audio_sha256'] or not
                (items[i]['real_request'] and items[j]['real_request'])):
            raise ValueError('paired audio identity differs')
    weights = logits.new_tensor([x['request_weight'] for x in items],dtype=torch.float32)
    targets = torch.stack([x['log_target'] for x in items]).to(device)
    mode=items[0].get('choice_mode','noul')
    if any(x.get('choice_mode','noul')!=mode for x in items):raise ValueError('incompatible task batch')
    joint = config.get('jev_joint_tpo')
    pair_counts = {}
    gap_counts = {}
    if joint:
        from .context_joint_tpo import view_loss
        value = view_loss(logits, targets, mode, joint)
        terms = {mode+'_joint_kl':value}
        value = value*.5
    elif mode in ('open','closed'):
        from .context_choice import projection as choice_projection
        value=choice_projection(logits,targets,torch.tensor([x['choice_gold'] for x in items],device=device),
                                mode,config.jev_choice)
        terms={mode+'_projection_kl':value}
    elif config.get('jev_reward_projection', False):
        from .context_tpo_rewards import projection
        value = projection(logits, targets)
        terms = dict(reward_projection_kl=value)
        gap_spec = config.get('jev_probability_gap')
        if gap_spec is not None:
            from .context_tpo_probability_gap import probability_gap
            penalty, gap_counts, diagnostics = probability_gap(
                logits, targets, torch.tensor([x['gold'] for x in items], device=device),
                torch.tensor([x['reference_rival'] for x in items], device=device), weights > 0, gap_spec)
            terms.update(diagnostics, probability_gap=penalty)
            value = value + penalty
        spec = config.get('jev_actor_pair_margin')
        if spec is not None:
            from .context_tpo_pair_margin import loss as pair_margin_loss
            ranking, pair_counts = pair_margin_loss(
                logits, targets,
                torch.tensor([x['gold'] for x in items], device=device),
                torch.tensor([x['pair_rival'] for x in items], device=device),
                torch.tensor([x['pair_enabled'] for x in items], device=device),
                weights > 0, spec,
            )
            terms['pair_margin'] = ranking
            value = value + ranking
    else:
        value, terms = objectives(logits, targets,
            torch.tensor([x['gold'] for x in items],device=device), torch.tensor(partners,device=device),
            torch.tensor(flags,device=device), Objective(**dict(config.jev_context_objective)),
            torch.tensor(confusion,device=device))
    # veRL divides each microbatch loss by accumulation; FSDP averages ranks.
    factor = (dist.get_world_size() * (config.ppo_mini_batch_size // len(items)) / items[0]['global_real'])
    total = (value * weights).sum() * factor
    metrics = {f'jev/{k}': float(((v * weights).sum() * factor).detach())
               for k, v in dict(terms, tpo_loss=value).items()}
    metrics.update({f'jev/pair_count/{k}': int(v) for k, v in pair_counts.items()})
    metrics.update({f'jev/gap_count/{k}': int(v) for k, v in gap_counts.items()})
    return total, metrics


EXTENSION = MODE, registered_loss, admit

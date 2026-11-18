"""Annexed curriculum source candidate over the admitted replica trainer.

No dispatch entry, no model/optimizer change and no package release. The
generated source reuses the frozen formal replica loop and swaps only the
registered loss adapter: the dual-credit behavior/rollout path remains intact,
while the loss is the pre-registered annealed curriculum. Every trajectory is
drawn once with Plackett-Luce without replacement; the same ordered draw is
used for ranking, calibration, contrast and the probability ratio.
"""
import ast
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from multimodal_jev.baseline import atomic_json
from multimodal_jev.context_curriculum_credit import (
    SCHEDULES, advantages, composite, schedule, trajectory,
)
from multimodal_jev.context_dual_credit import verify
from multimodal_jev.context_topology_batch_plan import derive
from tools.reports.context_replica_formal_candidate import (
    generate as formal_source, validate_schedule, window,
)


CURRICULUM_ALGORITHM = 'curriculum_grpo_floor_clipped_v1'
BEHAVIOR_ALGORITHM = 'dual_credit_grpo_floor_clipped_v1'
SFT_SPLIT_KEYS = {'nll', 'infonce'}
CONSTANT_KEYS = {'group_size', 'trajectory_k', 'margin', 'contrast_fraction',
                 'std_floor', 'clip_ratio', 'dual_clip_c', 'seed', 'sft_split'}
_CONFIG = None
_LAST_DIAGNOSTICS = None


def _constants(config):
    constants = config.get('constants')
    if not isinstance(constants, dict) or set(constants) != CONSTANT_KEYS:
        raise ValueError('complete curriculum constants required')
    if (type(constants['group_size']) is not int or constants['group_size'] < 2
            or type(constants['trajectory_k']) is not int or constants['trajectory_k'] < 1
            or type(constants['seed']) is not int or constants['seed'] < 0
            or type(constants['margin']) not in (int, float)
            or not math.isfinite(constants['margin']) or not 0 <= constants['margin'] < 1
            or type(constants['contrast_fraction']) not in (int, float)
            or not math.isfinite(constants['contrast_fraction'])
            or not 0 < constants['contrast_fraction'] < 1
            or type(constants['std_floor']) not in (int, float)
            or not math.isfinite(constants['std_floor']) or constants['std_floor'] < 0
            or type(constants['clip_ratio']) not in (int, float)
            or not math.isfinite(constants['clip_ratio'])
            or not 0 < constants['clip_ratio'] < 1
            or type(constants['dual_clip_c']) not in (int, float)
            or not math.isfinite(constants['dual_clip_c']) or constants['dual_clip_c'] <= 1):
        raise ValueError('registered curriculum constants required')
    split = constants['sft_split']
    if (not isinstance(split, dict) or set(split) != SFT_SPLIT_KEYS
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                   for v in split.values())
            or abs(sum(split.values()) - 1.0) > 1e-12):
        raise ValueError('declared convex NLL/InfoNCE split required')
    return dict(constants, sft_nll=float(split['nll']), sft_infonce=float(split['infonce']))


def configure(config):
    """Bind the declared schedule and loss constants for the generated trainer."""
    global _CONFIG
    reset_diagnostics()
    plan = derive(config['batch_config'])
    constants = _constants(config)
    total = config.get('schedule', {}).get('total_updates')
    if total is not None and total != plan['total_updates']:
        raise ValueError('declared schedule length differs from topology')
    _CONFIG = dict(constants, total=plan['total_updates'])
    return dict(_CONFIG)


def reset_diagnostics():
    global _LAST_DIAGNOSTICS
    _LAST_DIAGNOSTICS = None


def contract(config):
    """Validate the pre-registered curriculum identity and return its topology."""
    plan = derive(config['batch_config'])
    steps = window(plan, config.get('evaluation_steps'))
    declared = config.get('schedule')
    if (config.get('experiment_kind') != 'formal_source'
            or type(config.get('start_update')) is not int or config.get('start_update') != 0
            or type(config.get('stop_update')) is not int
            or config.get('stop_update') != plan['total_updates']
            or config.get('stage') != 'task' or config.get('resume_root')
            or config.get('engineering_prefix') is not False
            or config.get('training_dispatch_allowed') is not True
            or any(type(s) is not int for s in config['evaluation_steps'])
            or config.get('seed') != 317 or type(config.get('seed')) is not int
            or config.get('algorithm') != CURRICULUM_ALGORITHM
            or config.get('algorithm_family') != BEHAVIOR_ALGORITHM
            or config.get('dual_arm') != 'joint'
            or not isinstance(declared, dict)
            or declared.get('name') not in SCHEDULES
            or declared.get('total_updates') != plan['total_updates']
            or declared.get('start') != [0.5, 0.3, 0.2]
            or declared.get('end') != [0.0, 0.4, 0.6]
            or declared.get('order') != ['supervised', 'contrast', 'ranking']):
        raise ValueError('fresh formal curriculum identity and declared curve required')
    constants = configure(config)
    rewards = config.get('rewards')
    declared_trajectory = config.get('trajectory')
    if (not isinstance(rewards, dict) or not isinstance(declared_trajectory, dict)
            or rewards.get('ranking', {}).get('trajectory_k') != constants['trajectory_k']
            or rewards.get('contrast', {}).get('trajectory_k') != constants['trajectory_k']
            or rewards.get('contrast', {}).get('margin') != constants['margin']
            or declared_trajectory.get('k') != constants['trajectory_k']
            or declared_trajectory.get('shared_by') != ['rank', 'brier', 'margin']):
        raise ValueError('reward and trajectory declarations must match the constants')
    return dict(experiment_kind='formal_source', updates=plan['total_updates'],
        world_size=plan['dp_replicas'], global_batch=plan['global_batch'],
        updates_per_epoch=plan['updates_per_epoch'],
        tail_real_queries=plan['tail_real_queries'],
        tail_padding_slots=plan['tail_padding_slots'], seed=317,
        evaluation_steps=list(steps),
        recovery_checkpoint_steps=steps[1:],
        behavior_export='every_update_before_dual_replica_ack',
        query_weights='uniform_real_zero_padding', source_selection_only=True,
        curriculum_algorithm=CURRICULUM_ALGORITHM,
        behavior_algorithm=BEHAVIOR_ALGORITHM,
        schedule=dict(declared), constants=dict(constants))


def topology(config, world, mounted):
    """Validate the admitted DP layout while delegating the behavior identity."""
    from tools.reports.context_replica_trainer_candidate import topology as replica_topology
    if config.get('algorithm') != CURRICULUM_ALGORITHM:
        raise ValueError('curriculum algorithm required for the declared topology')
    return replica_topology(dict(config, algorithm=BEHAVIOR_ALGORITHM), world, mounted)


def loss_coefficients(update, total, *, contrast_fraction=0.5):
    """Effective coefficient on every differentiable or sampled loss term."""
    weights = schedule(update, total)
    return dict(supervised=weights['supervised'], contrast=weights['contrast'],
        ranking=weights['ranking'],
        margin=weights['contrast'] * contrast_fraction,
        brier=weights['contrast'] * (1.0 - contrast_fraction),
        rank=weights['ranking'])


def current_log_prob(weights, draws):
    """Differentiable FP64 log-probability of the exact same ordered draw.

    The behavior capture uses the detached exact helper in the credit module.
    This function repeats the same arithmetic on current weights without
    resampling, so the ratio compares two evaluations of one trajectory.
    """
    _trajectory_weights(weights, 'current weights')
    if (not isinstance(draws, torch.Tensor) or draws.dtype != torch.long
            or draws.ndim < 1 or draws.shape[-1] < 1):
        raise ValueError('long ordered candidate indices required')
    if draws.numel() == 0:
        return torch.empty(draws.shape[:-1], dtype=torch.float64, device=weights.device)
    if int(draws.min()) < 0 or int(draws.max()) >= len(weights):
        raise ValueError('draw indices inside the menu required')
    w = weights.double()
    rows = draws.reshape(-1, draws.shape[-1])
    if any(len(torch.unique(row)) != row.numel() for row in rows):
        raise ValueError('distinct candidates inside each ordered draw required')
    terms = []
    for row in rows:
        available = list(range(len(w)))
        value = torch.zeros((), dtype=torch.float64, device=w.device)
        for index in row.tolist():
            value = value + torch.log(w[index] / w[available].sum())
            available.remove(index)
        terms.append(value)
    return torch.stack(terms).reshape(draws.shape[:-1])


def _trajectory_weights(weights, name='weights'):
    if (not isinstance(weights, torch.Tensor) or weights.ndim != 1
            or weights.dtype != torch.float32 or not torch.isfinite(weights).all()
            or ((weights <= 0) | (weights >= 1)).any()):
        raise ValueError('finite interior FP32 '+name+' required; no clipping')
    return weights


def trajectory_terms(weights, labels, *, schedule_weights, k, group_size, seed,
                     update, query_id, margin=0.2, contrast_fraction=0.5,
                     std_floor=0.25):
    """One shared draw plus its rewards, composite score and GRPO advantages."""
    out = trajectory(weights, labels, k=k, group_size=group_size, margin=margin,
        seed=seed, update=update, query_id=query_id)
    scores = composite(dict(weights=schedule_weights, margin=out['margin'],
        brier=out['brier'], rank=out['rank']), contrast_fraction=contrast_fraction)
    group_advantages = advantages(scores.unsqueeze(0), floor=std_floor)[0]
    return dict(out, scores=scores, advantages=group_advantages,
                schedule_weights=dict(schedule_weights))


def surrogate_loss(current, behavior, group_advantages, *, clip_ratio=0.2,
                   dual_clip=3.0):
    """Ratio-clipped policy surrogate on one fixed trajectory."""
    for value, name in ((current, 'current log-prob'), (behavior, 'behavior log-prob'),
                        (group_advantages, 'group advantages')):
        if (not isinstance(value, torch.Tensor) or value.ndim != 1
                or value.dtype not in (torch.float32, torch.float64)
                or not torch.isfinite(value).all()):
            raise ValueError('finite one-dimensional '+name+' required')
    if current.shape != behavior.shape or current.shape != group_advantages.shape:
        raise ValueError('matching current/behavior/advantage vectors required')
    if (type(clip_ratio) not in (int, float) or not 0 < clip_ratio < 1
            or type(dual_clip) not in (int, float) or dual_clip <= 1):
        raise ValueError('declared clipped surrogate constants required')
    ratio = torch.exp(current.double() - behavior.double())
    unclipped = ratio * group_advantages.double()
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * group_advantages.double()
    objective = torch.min(unclipped, clipped)
    objective = torch.where(group_advantages.double() < 0,
        torch.max(objective, dual_clip * group_advantages.double()), objective)
    return -objective.mean(), dict(ratio_mean=float(ratio.detach().mean()),
        clip_fraction=float(((ratio < 1.0 - clip_ratio) | (ratio > 1.0 + clip_ratio))
                            .float().mean()))


def supervised_loss(logits, labels, *, nll_weight, infonce_weight):
    """The only differentiable data term: binary NLL plus listwise InfoNCE."""
    z = logits.float()
    if (z.ndim != 1 or not torch.isfinite(z).all()
            or not isinstance(labels, torch.Tensor) or labels.dtype != torch.bool
            or labels.shape != z.shape):
        raise ValueError('finite complete candidate logits and Boolean labels required')
    labels = labels.to(z.device)
    if (type(nll_weight) not in (int, float) or type(infonce_weight) not in (int, float)
            or not math.isfinite(nll_weight) or not math.isfinite(infonce_weight)
            or nll_weight < 0 or infonce_weight < 0):
        raise ValueError('declared nonnegative NLL/InfoNCE split required')
    nll = F.binary_cross_entropy_with_logits(z, labels.float(), reduction='mean')
    if bool(labels.any()):
        infonce = torch.logsumexp(z, 0) - torch.logsumexp(z[labels], 0)
    else:
        infonce = z.sum() * 0
    return nll_weight * nll + infonce_weight * infonce, dict(nll=nll, infonce=infonce)


def kl_loss(logits, reference_probability):
    """Fixed-reference binary KL, matching the v9 dual-credit regularizer."""
    z = logits.float()
    reference = reference_probability.float().to(z.device)
    if (z.ndim != 1 or not torch.isfinite(z).all()
            or reference.shape != z.shape or not torch.isfinite(reference).all()
            or reference.requires_grad
            or ((reference <= 0) | (reference >= 1)).any()):
        raise ValueError('complete logits and detached interior reference required')
    probability = z.sigmoid()
    if bool(((probability <= 0) | (probability >= 1)).any()):
        raise ValueError('interior current probabilities required for the KL')
    current = torch.stack((1.0 - probability, probability), -1)
    fixed = torch.stack((1.0 - reference, reference), -1)
    return (current * (torch.log(current) - torch.log(fixed))).sum()


def global_reduce(local_sum, *, world_size, global_real_requests):
    """Scale a local real-query sum to the global real-query denominator."""
    if (not isinstance(local_sum, torch.Tensor)
            or type(world_size) is not int or world_size < 1
            or type(global_real_requests) is not int or global_real_requests < 1):
        raise ValueError('tensor local sum and positive global denominators required')
    return local_sum * world_size / global_real_requests


def _arg(name):
    return 'curriculum integration anchor changed: ' + name


def generate(source_root):
    """Return the generated trainer source; this function does not run it."""
    source = formal_source(source_root)
    replacements = [
        ("    from tools.reports.context_dual_actor_candidate import load as load_dual_actor\n"
         "    dual_actor=load_dual_actor(c['source_root'])",
         "    import tools.reports.context_dual_actor_candidate as _dual_actor_module\n"
         "    from tools.reports.context_curriculum_candidate import configure as _configure_curriculum,hybrid_loss as _curriculum_hybrid_loss\n"
         "    _configure_curriculum(c)\n"
         "    _dual_actor_module.hybrid_loss=_curriculum_hybrid_loss\n"
         "    from tools.reports.context_dual_actor_candidate import load as load_dual_actor\n"
         "    dual_actor=load_dual_actor(c['source_root'])"),
        ("        from tools.reports.context_replica_formal_candidate import contract as formal_contract,validate_schedule\n"
         "        formal=formal_contract(c)",
         "        from tools.reports.context_curriculum_candidate import contract as curriculum_contract,validate_schedule\n"
         "        formal=curriculum_contract(c)"),
        ("    from tools.reports.context_replica_trainer_candidate import topology\n",
         "    from tools.reports.context_curriculum_candidate import topology\n"),
        ("        options=dict(algorithm=c['algorithm'],arm=c['dual_arm'],ranking_coefficient=.1,probability_coefficient=.3,kl_coefficient=.01,reference_policy_id=c['source_policy_id'])",
         "        options=dict(algorithm='"+BEHAVIOR_ALGORITHM+"',arm=c['dual_arm'],ranking_coefficient=.1,probability_coefficient=.3,kl_coefficient=.01,reference_policy_id=c['source_policy_id'])"),
        ("                    audio_token_id=base.config.audio_token_id,algorithm=c['algorithm'],start_update=start_update)",
         "                    audio_token_id=base.config.audio_token_id,algorithm='"+BEHAVIOR_ALGORITHM+"',start_update=start_update)"),
        ("objective=dict(group_size=16,reward='component_binary_correctness',kl=.01,auxiliary='constant_pair0.1_nll0.3')",
         "objective=dict(group_size=16,reward='curriculum_shared_trajectory',kl=.01,auxiliary='annealed_nll_infonce')"),
        ("                metrics=actor.update_policy(batch);torch.npu.synchronize();seconds=time.monotonic()-started",
         "                metrics=actor.update_policy(batch);torch.npu.synchronize();seconds=time.monotonic()-started\n"
         "                from tools.reports.context_curriculum_candidate import commit_diagnostics\n"
         "                commit_diagnostics(out,index)"),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError(_arg(before[:90]))
        source = source.replace(before, after)
    ast.parse(source)
    return source


def run(config, output):
    """Execute the generated candidate; no independent dispatch is started."""
    contract(config)
    namespace = {'__name__': __name__ + '.derived'}
    exec(compile(generate(config['source_root']), '<curriculum>', 'exec'), namespace)
    return namespace['run'](config, Path(output))


def _reward_stats(values):
    mean = float(values.mean())
    std = float(values.std(unbiased=False))
    return dict(mean=mean, std=std, no_signal=std == 0.0)


def hybrid_loss(logits, records, identities, reference_probability, real_requests, *,
                global_real_requests, world_size=1, arm='joint'):
    """Curriculum objective with the same call signature as the dual adapter."""
    global _LAST_DIAGNOSTICS
    if _CONFIG is None:
        raise ValueError('curriculum must be configured before the loss')
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
            or not torch.isfinite(logits).all()
            or len(records) != logits.shape[0] or len(identities) != len(records)
            or not isinstance(reference_probability, torch.Tensor)
            or reference_probability.shape != logits.shape
            or reference_probability.requires_grad
            or reference_probability.device.type != 'cpu'
            or real_requests.dtype != torch.bool or real_requests.shape != (len(records),)
            or type(world_size) is not int or world_size < 1
            or type(global_real_requests) is not int
            or global_real_requests < max(1, int(real_requests.sum()))):
        raise ValueError('complete curriculum batch and global denominator required')
    total = logits.float().sum() * 0
    rl_sum = logits.float().sum() * 0
    sft_sum = logits.float().sum() * 0
    kl_sum = logits.float().sum() * 0
    queries = []
    no_signal = set()
    representative = None
    for i, (record, identity) in enumerate(zip(records, identities, strict=True)):
        if not bool(real_requests[i]):
            continue  # zero-weight padding never enters a numerator or denominator
        binding = verify(record)['binding']
        if (set(identity) != {'policy_id', 'input_id', 'query_id', 'update'}
                or any(binding[k] != identity[k] for k in identity)):
            raise ValueError('stale curriculum behavior/input/update')
        if binding['algorithm'] != BEHAVIOR_ALGORITHM:
            raise ValueError('curriculum requires the registered behavior algorithm')
        if binding['group_size'] != _CONFIG['group_size'] or binding['seed'] != _CONFIG['seed']:
            raise ValueError('curriculum group size and seed differ from the declaration')
        candidates = binding['candidates']
        if _CONFIG['trajectory_k'] > len(candidates):
            raise ValueError('trajectory k must not exceed the candidate menu')
        labels = torch.tensor(binding['labels'], dtype=torch.bool)
        behavior_logits = torch.tensor(binding['logits'], dtype=torch.float32)
        old_weights = torch.softmax(behavior_logits, 0)
        _trajectory_weights(old_weights, 'behavior weights')
        current_weights = torch.softmax(logits[i].float(), 0)
        _trajectory_weights(current_weights, 'current weights')
        weights = schedule(identity['update'], _CONFIG['total'])
        representative = weights
        terms = trajectory_terms(old_weights, labels, schedule_weights=weights,
            k=_CONFIG['trajectory_k'], group_size=_CONFIG['group_size'],
            seed=_CONFIG['seed'], update=identity['update'], query_id=identity['query_id'],
            margin=_CONFIG['margin'], contrast_fraction=_CONFIG['contrast_fraction'],
            std_floor=_CONFIG['std_floor'])
        behavior_lp = terms['log_prob'].detach()
        current_lp = current_log_prob(current_weights, terms['draws'])
        if bool((behavior_lp > 0).any()) or bool((current_lp > 0).any()):
            raise ValueError('trajectory log-probabilities must be nonpositive')
        rl, ratio_detail = surrogate_loss(current_lp, behavior_lp.to(current_lp.device),
            terms['advantages'].to(current_lp.device),
            clip_ratio=_CONFIG['clip_ratio'], dual_clip=_CONFIG['dual_clip_c'])
        rl = rl.float()
        sft, sft_detail = supervised_loss(logits[i].float(), labels,
            nll_weight=_CONFIG['sft_nll'], infonce_weight=_CONFIG['sft_infonce'])
        kl = kl_loss(logits[i], reference_probability[i]).float()
        total = total + rl + weights['supervised'] * sft + 0.01 * kl
        rl_sum = rl_sum + rl
        sft_sum = sft_sum + weights['supervised'] * sft
        kl_sum = kl_sum + kl
        stats = {name: _reward_stats(terms[name]) for name in ('rank', 'brier', 'margin')}
        for name, value in stats.items():
            if value['no_signal']:
                no_signal.add(name)
        log_prob_std = float(behavior_lp.std(unbiased=False))
        if log_prob_std == 0.0:
            no_signal.add('trajectory_log_prob')
        reasons = {}
        for reason in terms['margin_reason']:
            reasons[reason] = reasons.get(reason, 0) + 1
        queries.append(dict(query_id=identity['query_id'], update=identity['update'],
            rewards=stats, log_prob=dict(mean=float(behavior_lp.mean()), std=log_prob_std),
            margin_reason=reasons, ratio=ratio_detail,
            nll=float(sft_detail['nll'].detach()), infonce=float(sft_detail['infonce'].detach()),
            supervised=float(weights['supervised'])))
    scale = global_reduce(torch.ones((), dtype=torch.float32, device=logits.device),
        world_size=world_size, global_real_requests=global_real_requests)
    total = total * scale
    rl_value = rl_sum * scale
    sft_value = sft_sum * scale
    kl_value = kl_sum * scale
    update = identities[0]['update'] if identities else None
    if _LAST_DIAGNOSTICS is None or _LAST_DIAGNOSTICS.get('update') != update:
        _LAST_DIAGNOSTICS = dict(update=update, schedule=None, queries=[],
            no_signal=set(), rl=0.0, supervised=0.0, kl_unweighted=0.0,
            total=0.0, micro_batches=0, world_size=world_size,
            global_real_requests=global_real_requests,
            scope='Loss interface only; no runtime or efficacy claim')
    accumulated = _LAST_DIAGNOSTICS
    if (accumulated['world_size'] != world_size
            or accumulated['global_real_requests'] != global_real_requests):
        raise ValueError('mixed global denominator within one update')
    accumulated['schedule'] = representative or accumulated['schedule']
    accumulated['queries'].extend(queries)
    accumulated['no_signal'].update(no_signal)
    accumulated['rl'] += float(rl_value.detach())
    accumulated['supervised'] += float(sft_value.detach())
    accumulated['kl_unweighted'] += float(kl_value.detach())
    accumulated['total'] += float(total.detach())
    accumulated['micro_batches'] += 1
    return total, dict(rl=float(rl_value.detach()), auxiliary=float(sft_value.detach()),
        kl=float(kl_value.detach()), supervised=float(sft_value.detach()),
        no_signal=sorted(no_signal), schedule=representative or accumulated['schedule'],
        arm=arm, scope='Loss interface only; no runtime or efficacy claim')


def commit_diagnostics(out, update):
    """Persist the accumulated loss diagnostics for one optimizer update."""
    if _LAST_DIAGNOSTICS is not None and _LAST_DIAGNOSTICS.get('update') == update:
        payload = dict(_LAST_DIAGNOSTICS)
        payload['no_signal'] = sorted(payload['no_signal'])
        atomic_json(Path(out) / f'curriculum-{update:04d}.json', payload)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run(json.loads(args.config.read_text()), args.output)

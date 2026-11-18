"""Detached loss accounting for a future simple-task actor, never a launcher.

All contributions use world_size/global_real_requests. Sum microsteps within
each rank, then average ranks. Never average the microstep list itself.
"""
import hashlib
import math
from pathlib import Path

ACTOR_SHA256 = 'a785016786955441b129b5432dec900259aaf80c197385f2e6773d563d865b29'
PREFIX = 'jev/components/'


def contributions(detail, payload, world_size):
    """Consume existing Python metrics; no tensor operation or model call."""
    n = payload['global_real_requests']
    if type(n) is not int or n < 1 or type(world_size) is not int or world_size < 1:
        raise ValueError('positive global denominator and world size required')
    stage = payload['stage']
    if stage not in ('instruction', 'task') or detail['stage'] != stage:
        raise ValueError('objective stage mismatch')
    aux = detail['auxiliary']
    if aux['global_real_requests'] != n:
        raise ValueError('auxiliary denominator mismatch')
    coefficients = [payload[k] for k in ('ranking_coefficient', 'probability_coefficient', 'kl_coefficient')]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in coefficients):
        raise ValueError('finite nonnegative coefficients required')
    ranking, probability, kl_weight = coefficients
    values = dict(ranking_unweighted=aux['ranking_numerator'] * world_size / n,
                  probability_unweighted=aux['probability_numerator'] * world_size / n,
                  rl=0., kl_unweighted=0.)
    if stage == 'task':
        sampled = detail['sampled']
        if (sampled['global_real_requests'] != n or sampled['coefficient'] != 0.
                or sampled['kl_coefficient'] != kl_weight):
            raise ValueError('sampled objective contract mismatch')
        # sampled rl and kl already include world_size/global_real_requests.
        values.update(rl=sampled['rl'], kl_unweighted=sampled['kl'])
    elif kl_weight != 0 or 'sampled' in detail:
        raise ValueError('instruction stage cannot contain sampled loss or KL')
    values.update(ranking_weighted=ranking * values['ranking_unweighted'],
                  probability_weighted=probability * values['probability_unweighted'],
                  kl_weighted=kl_weight * values['kl_unweighted'])
    values['reconstructed_total'] = sum(values[k] for k in
        ('rl', 'ranking_weighted', 'probability_weighted', 'kl_weighted'))
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values.values()):
        raise ValueError('finite detached scalar metrics required')
    return {PREFIX + k: v for k, v in values.items()}


def validate_total(metrics):
    """FP32 rounding may differ because components were converted separately."""
    total = metrics['jev/context_loss']
    magnitude = sum(abs(metrics[PREFIX + k]) for k in
                    ('rl', 'ranking_weighted', 'probability_weighted', 'kl_weighted'))
    if (not math.isfinite(total) or
            abs(total - metrics[PREFIX + 'reconstructed_total']) > 2e-7 + 2e-6 * magnitude):
        raise ValueError('component sum differs from actual mixed loss')


def generate(actor_path):
    """Derive a separate actor source from the admitted arithmetic-v2 package.

Does not import, register or alter the original actor or any training package.
"""
    raw = Path(actor_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ACTOR_SHA256:
        raise ValueError('admitted actor identity changed')
    text = raw.decode()
    replacements = [
        ("MODE='jev_context_hybrid_v1'", "MODE='jev_context_components_candidate_v1'"),
        ("ContextVar('jev_context_hybrid_active'", "ContextVar('jev_context_components_candidate_active'"),
        ('        value,_=loss(', '        value,detail=loss('),
        ('        total=total+value\n',
         '        total=total+value\n'
         '        for key, scalar in contributions(detail,p,dist.get_world_size()).items():\n'
         '            metrics[key]=metrics.get(key,0.)+scalar\n'),
        ("    metrics['jev/context_loss']=float(total.detach())\n",
         "    metrics['jev/context_loss']=float(total.detach())\n    validate_total(metrics)\n"),
    ]
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError('actor transformation anchor changed')
        text = text.replace(old, new)
    text = ('from tools.reports.context_loss_metrics_candidate import contributions, validate_total\n' + text)
    compile(text, '<context-loss-metrics-candidate>', 'exec')
    return text, dict(source_sha256=ACTOR_SHA256,
        generated_sha256=hashlib.sha256(text.encode()).hexdigest(),
        reduction='sum_microsteps_per_rank_then_mean_ranks',
        scope='Candidate source only; not installed or NPU validated')

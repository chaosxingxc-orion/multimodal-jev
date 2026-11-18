"""CPU analysis only: exact expected singleton F1 and its independent-logit gradient.

Reverse the Poisson-binomial recurrence in O(C^2), with no sampled actions.
This is not the GRPO normalized/clipped objective and is not wired to training.
"""
import math


def expected_f1_gradient(probabilities, gold):
    if (not probabilities or type(gold) is not int or not 0 <= gold < len(probabilities)
            or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                   for p in probabilities)):
        raise ValueError('Finite complete Bernoulli probabilities and single gold index required')
    indices = [i for i in range(len(probabilities)) if i != gold]
    tape = [[1.0]]
    for i in indices:
        p = probabilities[i]; previous = tape[-1]
        current = [0.0] * (len(previous) + 1)
        for k, mass in enumerate(previous):
            current[k] += (1 - p) * mass
            current[k + 1] += p * mass
        tape.append(current)
    coefficients = [2 / (k + 2) for k in range(len(tape[-1]))]
    conditional = sum(c * mass for c, mass in zip(coefficients, tape[-1]))
    value = probabilities[gold] * conditional
    dp = [0.0] * len(probabilities); dp[gold] = conditional
    adjoint = [probabilities[gold] * c for c in coefficients]
    for t in range(len(indices) - 1, -1, -1):
        i = indices[t]; p = probabilities[i]; previous = tape[t]
        dp[i] = sum(mass * (adjoint[k + 1] - adjoint[k]) for k, mass in enumerate(previous))
        adjoint = [(1 - p) * adjoint[k] + p * adjoint[k + 1] for k in range(len(previous))]
    return dict(expected_f1=value, probability_gradient=dp,
                logit_gradient=[g * p * (1 - p) for g, p in zip(dp, probabilities)],
                scope='Raw expected reward, complete singleton gold, factorized Bernoulli. '
                      'Probability endpoints use limiting sigmoid-logit gradient. No GRPO normalization or clipping.')

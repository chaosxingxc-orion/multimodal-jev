"""Verifiable single-decision reward; no self-reported execution reward."""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
from .task_schema import JevExample


@dataclass(frozen=True)
class RewardWeights:
    semantic: float = 1.0
    schema_validity: float = 0.1
    calibration: float = 0.1
    abstention: float = 0.1


def score_rollout(example: JevExample, prediction, *, confidence=None,
                  weights=RewardWeights()):
    valid = example.schema.accepts(prediction)
    correct = valid and type(prediction) is type(example.target) and prediction == example.target
    # Numeric levels may be serialized as float or int; booleans are excluded.
    if example.schema.decision_type == 'score' and valid:
        correct = prediction == example.target
    calibration = 0.0
    if confidence is not None:
        if type(confidence) not in (float, int) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('confidence must be finite and in [0, 1]')
        calibration = -(confidence - float(correct)) ** 2
    abstention = float(prediction is None and example.target is None) - float(prediction is None and example.target is not None)
    parts = {'semantic': float(correct), 'schema_validity': 0.0 if valid else -1.0,
             'calibration': calibration, 'abstention': abstention}
    parts['total'] = sum(getattr(weights, key) * value for key, value in parts.items())
    return parts


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """verl custom_reward_function callable, returning a scalar.

    ground_truth is JSON encoding of a full JevExample. It stays in the reward
    column of the training table, never in a prompt or rollout request.
    """
    example = JevExample.from_record(json.loads(ground_truth))
    try:
        answer = json.loads(solution_str)
        if not isinstance(answer, dict) or set(answer) - {'value', 'confidence'} or 'value' not in answer:
            return -1.0
        return score_rollout(example, answer['value'], confidence=answer.get('confidence'))['total']
    except (ValueError, TypeError):
        return -1.0

"""Versioned thinking requirements; no textual rationale or implicit controller.

The current runtime supports requirement-conditioned single-pass decisions only.
A valid multi-round specification is not permission to silently run one round.
"""
from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class ThinkingSpec:
    version: str
    requirements: tuple[str, ...]
    mode: str = 'direct'
    max_rounds: int = 1
    output: str = 'decision_only'

    def __post_init__(self):
        if type(self.version) is not str or not self.version.strip():
            raise ValueError('thinking version required')
        if (type(self.requirements) is not tuple or not self.requirements
                or any(type(r) is not str or not r.strip() for r in self.requirements)
                or len(set(self.requirements)) != len(self.requirements)):
            raise ValueError('distinct nonempty thinking requirements required')
        if self.mode not in ('direct', 'bounded_review'):
            raise ValueError('unsupported thinking mode')
        if type(self.max_rounds) is not int or self.max_rounds < 1:
            raise ValueError('positive integer decision-round budget required')
        if self.mode == 'direct' and self.max_rounds != 1:
            raise ValueError('direct mode requires exactly one round')
        if self.output != 'decision_only':
            raise ValueError('thinking contract outputs decisions only')

    @classmethod
    def from_dict(cls, value):
        fields = {'version', 'requirements', 'mode', 'max_rounds', 'output'}
        if type(value) is not dict or set(value) != fields:
            raise ValueError('exact thinking specification fields required')
        if type(value['requirements']) not in (tuple, list):
            raise ValueError('thinking requirements must be a sequence')
        return cls(**{**value, 'requirements': tuple(value['requirements'])})

    def to_dict(self):
        return {**asdict(self), 'requirements': list(self.requirements)}


def direct_suffix(value):
    """Fail closed when a caller has no implementation for requested review."""
    if value is None:
        return ''
    spec = value if isinstance(value, ThinkingSpec) else ThinkingSpec.from_dict(value)
    if spec.mode != 'direct':
        raise ValueError('bounded review requires an implemented decision controller')
    return ('\nThinking requirements (apply to the audio decision; output only the '
            'requested decision, without a textual reasoning trace):\n'
            + json.dumps(spec.to_dict(), ensure_ascii=False, sort_keys=True))

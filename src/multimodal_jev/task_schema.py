"""Serializable decision contracts. Targets never belong in model input."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import math
from .thinking import ThinkingSpec


@dataclass(frozen=True)
class Candidate:
    id: str
    description: str

    def __post_init__(self):
        if not self.id or not self.description:
            raise ValueError('candidate id and description are required')


@dataclass(frozen=True)
class DecisionSchema:
    id: str
    decision_type: str
    question: str
    candidates: tuple[Candidate, ...] = ()
    allow_unknown: bool = True
    levels: tuple[float, ...] = ()

    def __post_init__(self):
        if not self.id or not self.question:
            raise ValueError('schema id and question are required')
        if self.decision_type not in {'choice', 'noul', 'score', 'open_slot', 'open_slots'}:
            raise ValueError('unknown decision type')
        if self.decision_type == 'choice' and len(self.candidates) < 2:
            raise ValueError('choice requires at least two candidates')
        if self.decision_type != 'choice' and self.candidates:
            raise ValueError('only choice uses candidates')
        if self.decision_type == 'score':
            if len(self.levels) < 2 or any(not math.isfinite(v) for v in self.levels):
                raise ValueError('score requires finite ordered levels')
            if tuple(sorted(set(self.levels))) != self.levels:
                raise ValueError('score levels must increase strictly')
        elif self.levels:
            raise ValueError('only score uses levels')
        ids = [c.id for c in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError('candidate ids must be unique')

    def accepts(self, value: Any) -> bool:
        # None means insufficient evidence, never the string label "none" in FSC.
        if value is None:
            return self.allow_unknown
        if self.decision_type == 'choice':
            return type(value) is str and value in {c.id for c in self.candidates}
        if self.decision_type == 'noul':
            return type(value) is bool
        if self.decision_type == 'score':
            return type(value) in (int, float) and math.isfinite(value) and value in self.levels
        if self.decision_type == 'open_slots':
            # Empty means no annotated occurrences, not unknown; keep duplicates.
            return type(value) is list and all(type(v) is str and bool(v.strip()) for v in value)
        return type(value) is str and bool(value.strip())

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value['candidates'] = tuple(Candidate(**c) for c in value.get('candidates', []))
        value['levels'] = tuple(value.get('levels', []))
        return cls(**value)


@dataclass(frozen=True)
class JevExample:
    example_id: str
    split: str
    audio: str
    schema: DecisionSchema
    target: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    thinking: ThinkingSpec | None = None

    def __post_init__(self):
        if self.thinking is not None and not isinstance(self.thinking, ThinkingSpec):
            raise ValueError("thinking must be a validated ThinkingSpec")
        if not self.schema.accepts(self.target):
            raise ValueError(f'invalid target for {self.schema.id}: {self.target!r}')
        if not self.example_id or not self.audio or self.split not in {'train', 'valid', 'test'}:
            raise ValueError('example id, audio, and official split are required')

    def to_record(self):
        record = asdict(self)
        if self.thinking is None:
            record.pop("thinking")
        else:
            record["thinking"] = self.thinking.to_dict()
        return record

    @classmethod
    def from_record(cls, record):
        value = {**record, 'schema': DecisionSchema.from_dict(record['schema'])}
        if 'thinking' in value:
            value['thinking'] = ThinkingSpec.from_dict(value['thinking'])
        return cls(**value)

    def model_input(self):
        value = {'audio': self.audio, 'schema': asdict(self.schema)}
        if self.thinking is not None:
            value['thinking'] = self.thinking.to_dict()
        return value


def render_schema(schema):
    import json
    return json.dumps(asdict(schema), ensure_ascii=False, sort_keys=True)

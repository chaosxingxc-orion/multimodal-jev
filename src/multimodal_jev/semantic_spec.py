"""Task-level semantic specifications compiled into existing typed Span inputs.

This compiler accepts definitions only: no audio, transcript, gold, difficulty,
or per-example routing. It adds no process targets, rewards or external loop.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from .span_contract import SpanRequest


def _texts(value, name):
    if (type(value) not in (list, tuple) or not value
            or any(type(x) is not str or not x.strip() for x in value)
            or len(set(value)) != len(value)):
        raise ValueError('distinct nonempty '+name+' required')
    return tuple(value)


@dataclass(frozen=True)
class SemanticSpec:
    version: str
    id: str
    objective: str
    evidence_rules: tuple[str, ...]
    binding_rules: tuple[str, ...]
    decision_pattern: tuple[str, ...]
    output_rule: str

    def __post_init__(self):
        if self.version != 'jev-semantic-spec-v1':
            raise ValueError('registered semantic specification version required')
        for name in ('id','objective','output_rule'):
            value=getattr(self,name)
            if type(value) is not str or not value.strip():
                raise ValueError('nonempty '+name+' required')
        for name in ('evidence_rules','binding_rules','decision_pattern'):
            if type(getattr(self,name)) is not tuple:
                raise ValueError('immutable rule tuples required')
            _texts(getattr(self,name),name)
        if '[JEV_SPAN_' in json.dumps(asdict(self)):
            raise ValueError('reserved readout marker in specification')

    @classmethod
    def from_dict(cls, value):
        fields={'version','id','objective','evidence_rules','binding_rules','decision_pattern','output_rule'}
        if type(value) is not dict or set(value)!=fields:
            raise ValueError('exact task-only semantic specification fields required')
        value=dict(value)
        for name in ('evidence_rules','binding_rules','decision_pattern'):
            value[name]=_texts(value[name],name)
        return cls(**value)

    def canonical(self):
        return json.dumps(asdict(self),ensure_ascii=False,sort_keys=True,separators=(',',':'))


@dataclass(frozen=True)
class DecisionDefinition:
    """Immutable task definition shared unchanged by every audio example."""
    questions: tuple
    final_id: str
    requirements: tuple[str, ...]
    semantic_spec_sha256: str

    def for_audio(self, audio):
        return SpanRequest(audio,self.questions,self.final_id,self.requirements)


def compile_definition(questions, final_id, spec):
    """Candidate definitions stay in schemas; spec supplies interpretation rules.

    This constructs model input only. Semantics of prose require task review;
    syntactic validity alone is not training admission or evidence of reasoning.
    """
    if not isinstance(spec,SemanticSpec):
        raise TypeError('validated SemanticSpec required')
    questions=tuple(questions)
    if len({q.schema.id for q in questions})!=len(questions):
        raise ValueError('distinct readout schema identities required')
    serialized=spec.canonical()
    requirements=(serialized,)
    # Reuse the actual readout contract validation, without real audio or gold.
    SpanRequest('/definition-validation-only.wav',questions,final_id,requirements)
    return DecisionDefinition(questions,final_id,requirements,
                              hashlib.sha256(serialized.encode()).hexdigest())

"""Input-only candidate-order interventions; no target or probability correction.

The stock encoder binds symbols by candidate position. Reordering therefore
changes both display order and symbol binding, not symbol identity in isolation.
"""
from dataclasses import replace
import hashlib
from .span_contract import SpanQuestion, SpanRequest
from .task_schema import DecisionSchema

VARIANTS = ('native', 'shift1', 'shift30', 'permutation317')


def from_input(value):
    if set(value) != {'audio', 'questions', 'final_id', 'requirements', 'version'}:
        raise ValueError('input-only request fields required')
    questions = []
    for q in value['questions']:
        if set(q) != {'id', 'schema'}:
            raise ValueError('input-only question fields required')
        questions.append(SpanQuestion(q['id'], DecisionSchema.from_dict(q['schema'])))
    return SpanRequest(value['audio'], tuple(questions), value['final_id'],
                       tuple(value['requirements']), value['version'])


def intervene(request, variant):
    if variant not in VARIANTS:
        raise ValueError('predefined input-independent intervention required')
    questions = []
    receipt = {}
    for q in request.questions:
        if q.schema.decision_type != 'choice' or q.schema.allow_unknown:
            raise ValueError('closed categorical support required')
        original = q.schema.candidates
        indices = list(range(len(original)))
        if variant.startswith('shift'):
            shift = int(variant[5:]) % len(indices)
            indices = indices[shift:] + indices[:shift]
        elif variant == 'permutation317':
            indices.sort(key=lambda i: hashlib.sha256(
                f'317:{q.id}:{original[i].id}'.encode()).hexdigest())
        candidates = tuple(original[i] for i in indices)
        questions.append(replace(q, schema=replace(q.schema, candidates=candidates)))
        receipt[q.id] = dict(canonical_values=list(q.values),
                             presented_values=[c.id for c in candidates],
                             presented_to_canonical=indices,
                             moved=sum(i != j for i, j in enumerate(indices)))
    return replace(request, questions=tuple(questions)), receipt


def canonicalize(values, mapping):
    """Restore semantic order without changing values (including logprobs)."""
    indices = mapping['presented_to_canonical']
    if len(values) != len(indices) or sorted(indices) != list(range(len(indices))):
        raise ValueError('complete bijective candidate mapping required')
    result = [None] * len(indices)
    for value, i in zip(values, indices, strict=True):
        result[i] = value
    return result

"""Dataset-independent logical Q operators over externally supplied concepts.

These construct finite predicates, not process gold. Caller selects a fixed
task-level menu before seeing examples and separately authenticates supervision.
"""
from dataclasses import asdict, dataclass
import json
from .span_contract import SpanQuestion
from .task_schema import DecisionSchema


@dataclass(frozen=True)
class Concept:
    id: str
    definition: str

    def __post_init__(self):
        if any(type(v) is not str or not v.strip() for v in (self.id,self.definition)):
            raise ValueError('concept identity and semantic definition required')
        if '[JEV_SPAN_' in self.id+self.definition:
            raise ValueError('reserved marker in concept')


def predicate(question_id, operator, *, concept=None, intent=None, slot=None, allow_unknown=False):
    """A semantic definition is a parameter, never a sample-dependent question.

    Multi-intent evidence and exclusive-label membership have different truth
    conditions. Existing single-intent labels only license the latter directly.
    """
    if operator in ('intent_expressed','intent_label_member','slot_type_present'):
        if not isinstance(concept,Concept) or intent is not None or slot is not None:
            raise ValueError('one concept definition required')
        definitions={'concept':asdict(concept)}
        prompts={
            'intent_expressed': 'Does the query express a user intention satisfying the supplied intent definition? Mere mention of its topic or an explicitly rejected action does not establish that intention.',
            'intent_label_member': 'Under the supplied single-intent task annotation policy, does the query belong to the supplied intent category? Judge task-category membership, not the absence of every other possible intention.',
            'slot_type_present': 'Does the query contain a semantic filler satisfying the supplied slot-type definition? Judge its role in context, not a matching word alone. Do not extract its value.',
        }
        prompt=prompts[operator]
    elif operator=='intent_slot_bound':
        if concept is not None or not isinstance(intent,Concept) or not isinstance(slot,Concept):
            raise ValueError('intent and slot definitions required')
        definitions={'intent':asdict(intent),'slot':asdict(slot)}
        prompt=('Does the query contain a filler of the defined slot type that serves the defined intention? '
                'Co-occurrence without semantic binding is insufficient. This predicate does not identify repeated intent instances.')
    else:
        raise ValueError('unregistered logical decision operator')
    if type(allow_unknown) is not bool:
        raise ValueError('explicit boolean unknown policy required')
    question=prompt+'\nDefinitions: '+json.dumps(definitions,ensure_ascii=False,sort_keys=True)
    return SpanQuestion(question_id,DecisionSchema(id='jev-predicate-v1:'+question_id,
        decision_type='noul',question=question,allow_unknown=allow_unknown))


@dataclass(frozen=True)
class PredicateMenu:
    """Fixed complete concept menu; no gold-aware sampling or routing."""
    operator: str
    concepts: tuple[Concept, ...]
    questions: tuple[SpanQuestion, ...]


def compile_menu(operator, concepts, *, allow_unknown=False):
    if operator not in ('intent_label_member','intent_expressed','slot_type_present'):
        raise ValueError('registered unary operator required')
    concepts=tuple(concepts)
    if (not concepts or any(not isinstance(c,Concept) for c in concepts)
            or len({c.id for c in concepts})!=len(concepts)):
        raise ValueError('distinct complete concept menu required')
    questions=tuple(predicate('q:'+c.id,operator,concept=c,allow_unknown=allow_unknown) for c in concepts)
    return PredicateMenu(operator,concepts,questions)


def predicate_targets(menu, *, positives, negatives, unavailable, label_basis):
    """Map independently authenticated annotations AFTER compiling input.

    Every concept needs explicit positive, negative or unavailable status. This
    function does not infer false from omission or certify semantic review.
    Masked targets use an inert false index, never an unknown semantic label.
    """
    if not isinstance(menu,PredicateMenu) or type(label_basis) is not str or not label_basis.strip():
        raise ValueError('fixed menu and explicit label basis required')
    groups=[]
    for values in (positives,negatives,unavailable):
        if type(values) not in (set,frozenset,list,tuple) or any(type(x) is not str for x in values):
            raise ValueError('explicit concept identities required')
        if len(set(values))!=len(values):
            raise ValueError('duplicate supervision identity')
        groups.append(set(values))
    pos,neg,missing=groups
    if pos&neg or pos&missing or neg&missing:
        raise ValueError('conflicting supervision states')
    ids={c.id for c in menu.concepts}
    if pos|neg|missing!=ids:
        raise ValueError('complete explicit supervision status for this menu required')
    return dict(operator=menu.operator,question_ids=[q.id for q in menu.questions],
        targets=[q.values.index(c.id in pos) for c,q in zip(menu.concepts,menu.questions,strict=True)],
        supervised=[c.id not in missing for c in menu.concepts],
        status=['positive' if c.id in pos else 'negative' if c.id in neg else 'unavailable' for c in menu.concepts],
        label_basis=label_basis)


def exclusive_class_targets(menu, label, *, label_basis):
    """Only an admitted exhaustive single-label classification projection."""
    if menu.operator!='intent_label_member':
        raise ValueError('exclusive class labels cannot certify semantic absence')
    ids={c.id for c in menu.concepts}
    if label not in ids:
        raise ValueError('class outside configured complete ontology')
    return predicate_targets(menu,positives={label},negatives=ids-{label},unavailable=set(),label_basis=label_basis)

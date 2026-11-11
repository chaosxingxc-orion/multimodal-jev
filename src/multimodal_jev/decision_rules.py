"""Explicit Boolean decision rules over reviewed facts, never raw label absence.

None is unavailable supervision here, not an actor abstention target. Three-value
composition can resolve a rule despite an irrelevant unknown branch; it never
fabricates false for a missing fact. Predicate definitions are model-visible;
source targets, internal rule IDs and dependency gold are not.
"""
from dataclasses import dataclass

from .task_schema import DecisionSchema
from .thinking import ThinkingSpec


@dataclass(frozen=True)
class Fact:
    value: bool | None
    status: str

    def __post_init__(self):
        if self.status not in ('reviewed', 'unavailable', 'conflict'):
            raise ValueError('explicit fact review status required')
        if self.status == 'reviewed':
            if type(self.value) is not bool:
                raise ValueError('reviewed Boolean required')
        elif self.value is not None:
            raise ValueError('unreviewed facts must not carry truth labels')


def validate(rule, definitions, depth=0):
    if depth > 16 or type(rule) is not dict:
        raise ValueError('bounded rule tree required')
    op = rule.get('op')
    if op == 'fact':
        if set(rule) != {'op', 'key'} or type(rule['key']) is not str or rule['key'] not in definitions:
            raise ValueError('known fact definition required')
        description = definitions[rule['key']]
        if type(description) is not str or not description.strip():
            raise ValueError('nonempty semantic definition required')
    elif op == 'not':
        if set(rule) != {'op', 'arg'}:
            raise ValueError('not requires one argument')
        validate(rule['arg'], definitions, depth+1)
    elif op in ('all', 'any'):
        if set(rule) != {'op', 'args'} or type(rule['args']) is not list or len(rule['args']) < 2:
            raise ValueError('all/any requires at least two arguments')
        for arg in rule['args']:
            validate(arg, definitions, depth+1)
    elif op == 'except':
        if set(rule) != {'op', 'base', 'exception'}:
            raise ValueError('except requires base and exception')
        validate(rule['base'], definitions, depth+1)
        validate(rule['exception'], definitions, depth+1)
    else:
        raise ValueError('unknown rule operator')
    return rule


def dependencies(rule):
    if rule['op'] == 'fact':
        return {rule['key']}
    if rule['op'] == 'not':
        return dependencies(rule['arg'])
    args = rule['args'] if rule['op'] in ('all','any') else [rule['base'],rule['exception']]
    return set().union(*(dependencies(arg) for arg in args))


def evaluate(rule, definitions, facts):
    validate(rule, definitions)
    if any(not isinstance(f, Fact) for f in facts.values()):
        raise ValueError('reviewed Fact records required')

    def neg(v):
        return None if v is None else not v

    def both(values):
        if False in values:
            return False
        return None if None in values else True

    def visit(node):
        op=node['op']
        if op=='fact':
            return facts.get(node['key'],Fact(None,'unavailable')).value
        if op=='not':
            return neg(visit(node['arg']))
        if op=='except':
            return both([visit(node['base']),neg(visit(node['exception']))])
        values=[visit(arg) for arg in node['args']]
        if op=='all':
            return both(values)
        if True in values:
            return True
        return None if None in values else False

    value=visit(rule)
    masked=sorted(k for k in dependencies(rule) if k not in facts or facts[k].status!='reviewed')
    return dict(value=value, supervision_status='available' if value is not None else 'unavailable',
                unresolved_fact_keys=masked, actor_abstention_label=False)


def family(rule, definitions):
    """Structural family, invariant to fact identity and commutative order.

    This is a grouping feature, not proof of independent semantic/grammar splits.
    Equivalent operators (e.g. except and all/not) still need common holdout
    assignment by the release's explicit family registry.
    """
    validate(rule, definitions)
    def shape(node):
        op=node['op']
        if op=='fact':return 'fact'
        if op=='not':return 'not('+shape(node['arg'])+')'
        if op=='except':return 'except('+shape(node['base'])+','+shape(node['exception'])+')'
        return op+'('+','.join(sorted(shape(x) for x in node['args']))+')'
    return shape(rule)


def question(rule, definitions):
    validate(rule, definitions)
    def render(node):
        op=node['op']
        if op=='fact':return '('+definitions[node['key']]+')'
        if op=='not':return 'NOT '+render(node['arg'])
        if op=='except':return '('+render(node['base'])+' AND NOT '+render(node['exception'])+')'
        return '('+(' AND ' if op=='all' else ' OR ').join(render(x) for x in node['args'])+')'
    return 'Using the complete audio evidence, decide whether this rule is true: '+render(rule)


def contract(rule, definitions, thinking):
    if not isinstance(thinking, ThinkingSpec):
        raise ValueError('explicit thinking requirements required')
    # Stable generic task ID cannot disclose rule family or supervision status.
    schema=DecisionSchema('configured-rule', 'noul', question(rule,definitions),allow_unknown=False)
    return schema, thinking

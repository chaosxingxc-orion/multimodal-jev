"""TRAIN-only descriptive strata; ontology ambiguity is not acoustic difficulty."""
from collections import Counter
import re


def ontology_strata(pairs):
    if not pairs or len({tuple(v) for v in pairs.values()}) != len(pairs):
        raise ValueError('unique canonical intent pairs required')
    if any(len(v) != 2 or any(not isinstance(s, str) or not s for s in v) for v in pairs.values()):
        raise ValueError('scenario/action pairs required')
    scenarios = Counter(v[0] for v in pairs.values())
    actions = Counter(v[1] for v in pairs.values())
    return {k: dict(scenario_competitors=scenarios[s], action_competitors=actions[a],
                   minimum_components=1 if scenarios[s] == 1 or actions[a] == 1 else 2)
            for k, (s, a) in pairs.items()}


def surface_flags(sentence, entity_types):
    """Offline reference-text triage only; never targets or model inputs.

    A trigger means review needed, not that a logical phenomenon is present.
    Entity counts describe existing annotations, not annotation completeness.
    """
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", sentence.casefold())
    words = set(tokens)
    return dict(
        token_count=len(tokens),
        short_reference=len(tokens) <= 3,
        long_reference=len(tokens) >= 15,
        negation_trigger=bool(words & {'not', 'never', 'without', 'except', "don't", "dont", "isn't", "can't"}),
        correction_trigger=bool(words & {'instead', 'rather', 'actually'}),
        connective_trigger=bool(words & {'and', 'or', 'then', 'before', 'after', 'if', 'unless'}),
        deictic_trigger=bool(words & {'it', 'this', 'that', 'there', 'them'}),
        annotated_entities=len(entity_types),
        annotated_types=len(set(entity_types)),
        repeated_annotated_type=len(entity_types) > len(set(entity_types)),
    )

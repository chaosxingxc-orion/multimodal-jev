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


def question_pack_profile(config, class_counts):
    """Describe label partitions, not model accuracy or reasoning depth.

    Gold projections are confined to this offline audit. In particular, this
    function must not select per-example input questions or prune candidates.
    """
    if not class_counts or any(type(n) is not int or n < 0 for n in class_counts.values()):
        raise ValueError('nonnegative integer counts for the complete ontology required')
    questions = {q['id']: q for q in config['questions']}
    if len(questions) != len(config['questions']):
        raise ValueError('duplicate question id')
    mappings = {}
    for qid, q in questions.items():
        if (q['default'] != 'outside_family'
                or q['values'] != list(q['class_sets']) + ['outside_family']):
            raise ValueError('explicit disjoint values and outside-family default required')
        mapping = {}
        for value, labels in q['class_sets'].items():
            if not labels:
                raise ValueError('empty partition')
            for label in labels:
                if label not in class_counts or label in mapping:
                    raise ValueError('unknown or multiply assigned intent')
                mapping[label] = value
        mappings[qid] = {label: mapping.get(label, 'outside_family') for label in class_counts}
    result = {}
    total = sum(class_counts.values())
    for name, ids in config['packs'].items():
        if len(ids) != 2 or len(set(ids)) != 2 or not set(ids) <= questions.keys():
            raise ValueError('two distinct known questions required')
        signatures = {k: tuple(mappings[q][k] for q in ids) for k in class_counts}
        sizes = Counter(signatures.values())
        component_sizes = [Counter(mappings[q].values()) for q in ids]
        family = [k for k, sig in signatures.items() if any(v != 'outside_family' for v in sig)]
        joint_only = [k for k in family if sizes[signatures[k]] == 1
                      and all(component_sizes[i][signatures[k][i]] > 1 for i in range(2))]
        single = [k for k in family if any(component_sizes[i][signatures[k][i]] == 1 for i in range(2))]
        unresolved = [k for k in family if sizes[signatures[k]] > 1]
        distributions = {}
        for q in ids:
            counts = Counter()
            for k, n in class_counts.items():
                counts[mappings[q][k]] += n
            distributions[q] = dict(counts)
        result[name] = dict(
            family_classes=family, family_recordings=sum(class_counts[k] for k in family),
            outside_recordings=total-sum(class_counts[k] for k in family),
            single_question_unique_classes=single,
            joint_only_unique_classes=joint_only,
            joint_only_recordings=sum(class_counts[k] for k in joint_only),
            unresolved_family_classes=unresolved,
            projected_answer_counts=distributions,
            constant_answer_accuracy={q: max(v.values())/total if total else None
                                      for q, v in distributions.items()},
        )
    return result

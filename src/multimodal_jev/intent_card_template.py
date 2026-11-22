"""Render an intent card bank with the established audio probability contract."""
from .context_candidate_task import compile_candidates


def compile_intent_card_request(audio, task, bank, template, arm, *, example_indices=None):
    options = template["arms"][arm]
    candidates = task["schema"]["candidates"]
    cards = {card["id"]: card for card in bank["cards"]}
    if len(cards) != len(bank["cards"]) or set(cards) != {row["id"] for row in candidates}:
        raise ValueError("one card per candidate is required")
    definitions = []
    for row in candidates:
        card = cards[row["id"]]
        text = row["description"]
        if options["definitions"]:
            text = template["boundary"].format(meaning=card["activation"])
        index = template["example_index"] if example_indices is None else example_indices[row["id"]]
        if options["examples"] and index is not None:
            example = card["examples"][index]["text"]
            text += template["example"].format(text=example)
        definitions.append(dict(id=row["id"], definition=text))
    rules = list(task["thinking"]["requirements"])
    if options["definitions"]:
        rules.append(template["boundary_rule"])
    if options["rules"]:
        rules.append(template["uncertainty"])
        for entry in bank["semantic_rules"]:
            rules.append(template["rule_format"].format(labels=", ".join(entry["labels"]), rule=entry["rule"]))
    if options["examples"]:
        rules.append(template["example_rule"])
    return compile_candidates(audio, definitions, rules, semantics="exclusive_event")

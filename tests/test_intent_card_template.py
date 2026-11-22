import json
from pathlib import Path

from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.context_decision_contract import render_prefix_context_request
from multimodal_jev.intent_card_template import compile_intent_card_request


def test_rule_switch_only_changes_rules_and_support_is_private():
    config = Path(__file__).resolve().parents[1] / "configs/portable/operations/intent-card-template-v1.json"
    template = json.loads(config.read_bytes())
    labels = ["up", "down"]
    task = {"schema": {"candidates": [{"id": k, "description": k} for k in labels]},
            "thinking": {"requirements": ["Use the audio."]}}
    bank = {"cards": [{"id": k, "activation": k, "examples": [{"text": "please " + k, "id": "private-id"}]}
                      for k in reversed(labels)],
            "semantic_rules": [{"labels": labels, "rule": "Use the requested direction."}]}
    arms = {arm: compile_intent_card_request("audio", task, bank, template, arm) for arm in template["arms"]}
    original = compile_candidates("audio", [dict(id=k, definition=k) for k in labels], ["Use the audio."],
                                  semantics="exclusive_event")
    assert arms["original"] == original
    plain, ruled = arms["definitions_example"], arms["definitions_rules_example"]
    assert plain.questions == ruled.questions and bank["semantic_rules"][0]["rule"] in ruled.requirements
    assert bank["semantic_rules"][0]["rule"] not in plain.requirements
    assert [q.id for q in ruled.questions] == labels
    assert [q.id for q in ruled.questions if "please " + q.id in q.definition] == labels
    assert len({render_prefix_context_request(r)[2] for r in arms.values()}) == 1
    assert "private-id" not in render_prefix_context_request(ruled)[0]

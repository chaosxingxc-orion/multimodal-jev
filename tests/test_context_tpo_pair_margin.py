import json

import pytest
import torch

from multimodal_jev.context_tpo_pair_margin import compile_pairs, loss, verify_update
from multimodal_jev.context_tpo_rewards import projection
from multimodal_jev.baseline import file_hash
from tools.reports.build_tpo_actor_pairs import cue_changes, source_text


def test_frozen_switch_uses_strict_reference_and_any_projected_rival():
    reference = torch.tensor([[.2, .7, .7], [.5, .5, .2], [.2, .7, .1], [.2, .7, .1]])
    projected = torch.tensor([[[.25, -.1, .2], [0., 1., 0.], [0., 1., 0.], [.25, 0., 0.]],
                              [[.5, 0., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]]])
    rivals, enabled = compile_pairs(reference, projected, torch.zeros(4, dtype=torch.long),
                                    torch.tensor([True, True, False, True]), .25)
    assert rivals.tolist() == [1, 1, 1, 1]
    assert enabled.tolist() == [[True, False, False, False], [False, False, False, True]]


def test_actor_gradient_and_padding_with_zero_weight_equivalence():
    x = torch.tensor([[0., .5, 0.], [0., .5, 0.], [.25, 0., 0.], [0., .5, 0.]], requires_grad=True)
    q = torch.tensor([[0., 1., 0.], [.5, 0., 0.], [0., 1., 0.], [0., 1., 0.]])
    gold, rival = torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.long)
    real = torch.tensor([True, True, True, False])
    baseline = (projection(x, q) * real).sum() / real.sum()
    gradients = torch.autograd.grad(baseline, x, retain_graph=True)[0]
    zero, counts = loss(x, q, gold, rival, torch.ones_like(real), real, dict(logit_margin=.25, weight=0.))
    zero_loss = baseline + zero.sum() / real.sum()
    assert torch.equal(baseline, zero_loss)
    assert torch.equal(gradients, torch.autograd.grad(zero_loss, x, retain_graph=True)[0])
    assert counts['active'].item() == 0
    added, counts = loss(x, q, gold, rival, torch.ones_like(real), real, dict(logit_margin=.25, weight=1/3))
    delta = torch.autograd.grad(baseline + added.sum() / real.sum(), x)[0] - gradients
    assert torch.allclose(delta, torch.tensor([[-1., 1., 0.], [-1., 1., 0.], [0., 0., 0.], [0., 0., 0.]]) / 9)
    assert dict((k, n.item()) for k, n in counts.items()) == dict(
        eligible=3, active=2, turned_positive=1, strengthened=1, still_nonpositive=0)


def test_tail_batch_receipt_requires_correct_denominator_and_counts():
    row = dict(step=2, epoch=0, real=3, metrics={
        'jev/reward_projection_kl': .125, 'jev/pair_margin': .0625, 'jev/tpo_loss': .1875},
        pair_margin=dict(eligible=2, active=2, turned_positive=0, strengthened=1, still_nonpositive=1))
    verify_update(row, [384, 3], [0, 0], [10, 2])
    row["pair_margin"]["active"] = 3
    with pytest.raises(ValueError, match="pair margin update"):
        verify_update(row, [384, 3], [0, 0], [10, 2])


def test_reference_text_screen_requires_independent_cue_and_recording_binding(tmp_path):
    labels = {"down": "quieter", "up": "louder"}
    cases = [
        ("match", "down", "play it quieter", "correct"),
        ("wrong", "up", "play it quieter", "correct"),
        ("multi", "up", "play it louder and then quieter", "correct"),
        ("negated", "down", "do not play it quieter", "correct"),
        ("unclear", "down", "music please", "correct"),
        ("bad_recording", "up", "play it louder", "incorrect"),
    ]
    metadata = tmp_path / "train.jsonl"
    metadata.write_text("".join(json.dumps(dict(intent=label, sentence=sentence, recordings=[
        dict(file=uid + ".flac", wer=0.25, status=status)])) + "\n" for uid, label, sentence, status in cases))
    eligible = {uid: dict(intent=label, audio=uid + ".flac", source_group=uid)
                for uid, label, *_ in cases}
    eligible["missing"] = dict(intent="up", audio="missing.flac", source_group="missing")
    config = dict(metadata=metadata, metadata_sha256=file_hash(metadata),
                  text_screen=dict(intent_cues={k: [v] for k, v in labels.items()}, exclude=r"\bnot\b",
                                   recording_status="correct"))
    text, rejects, decisions = source_text(config, eligible)
    assert set(text) == {"match"}
    assert text["match"]["cue_spans"] == [(2, 3)]
    assert decisions["wrong"] == "intent_cue_disagrees_with_label"
    assert decisions["multi"] == "conflicting_intent_cues"
    assert decisions["missing"] == "missing_recording_binding"
    assert sum(rejects.values()) + len(text) == len(eligible)


def test_pair_requires_changed_tokens_to_overlap_intent_evidence_on_both_sides():
    play_faster = {"words": ("play", "it", "faster"), "cue_spans": [(2, 3)]}
    play_slower = {"words": ("play", "it", "slower"), "cue_spans": [(2, 3)]}
    assert cue_changes(play_faster, play_slower, (("play", "it"), ()))
    tomorrow = {"words": ("play", "it", "faster", "tomorrow"), "cue_spans": [(2, 3)]}
    today = {"words": ("play", "it", "faster", "today"), "cue_spans": [(2, 3)]}
    assert not cue_changes(tomorrow, today, (("play", "it", "faster"), ()))

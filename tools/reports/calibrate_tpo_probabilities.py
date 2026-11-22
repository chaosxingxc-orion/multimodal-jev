"""Fit and apply a harness threshold using only a frozen source TRAIN roster."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json


def source_panel(spec, parent):
    from multimodal_jev.context_candidate_task import compile_candidates
    from tools.reports.run_native_seal_endpoint import bound

    roster = bound(spec["roster"])
    release = Path(parent["release"])
    manifest = bound(dict(path=release / "MANIFEST.json", sha256=parent["release_sha256"]))
    task = bound(dict(path=release / "tasks.json", sha256=manifest["files"]["tasks.json"]))["intent"]
    candidates = [dict(id=row["id"], definition=row["description"]) for row in task["schema"]["candidates"]]
    ids = [row["id"] for row in candidates]
    learning = release / "learning.jsonl"
    if file_hash(learning) != manifest["files"]["learning.jsonl"]:
        raise ValueError("source TRAIN release changed")
    requested = {row["id"]: row for row in roster}
    if len(requested) != len(roster) or len(roster) != spec["recordings"]:
        raise ValueError("incomplete calibration population")
    rows = {}
    with learning.open() as stream:
        for line in stream:
            row = json.loads(line)
            uid = row["id"]
            if uid not in requested:
                continue
            if (uid in rows or row["official_split"] != "train" or row["role"] != "learning"
                    or row["known_pending_source"] is not False
                    or any(row[key] != value for key, value in requested[uid].items())):
                raise ValueError("calibration row differs from the clean TRAIN release")
            rows[uid] = row
    counts = Counter(row["intent"] for row in roster)
    if (set(rows) != set(requested) or set(counts) - set(ids)
            or [key for key in ids if not counts[key]] != spec["absent_classes"]):
        raise ValueError("calibration population or missing class changed")
    assets = [dict(id=uid, audio_sha256=rows[uid]["audio_sha256"]) for uid in requested]

    def request(asset):
        return compile_candidates(rows[asset["id"]]["audio"], candidates,
                                  task["thinking"]["requirements"], semantics="exclusive_event")

    return roster, assets, ids, request


def cached_source(spec, parent):
    from multimodal_jev.cached_context_preparation import CachedContextPreparer, processor_cache_identity
    from multimodal_jev.readonly_encoded_audio_cache import ReadOnlyEncodedAudioCache
    from tools.reports.context_cached_reference_trainer import no_online_processor

    _, assets, ids, request = source_panel(spec, parent)
    prepare = CachedContextPreparer(
        no_online_processor, ReadOnlyEncodedAudioCache(parent["encoded_audio_cache"]),
        processor_sha256=processor_cache_identity(parent["source_policy"]),
        decision_token_id=parent["source_contract"]["token"]["id"],
        audio_token_id=parent["audio_token_id"], max_tokens=parent["view_execution"]["max_input_tokens"],
        prompt_layout=parent["prompt_layout"],
    )

    def encode(asset):
        value = prepare((request(asset), asset["audio_sha256"]))
        return dict(inputs=value.inputs, positions=value.positions), None

    return assets, ids, encode


def thresholds(probabilities, gold):
    raw = np.asarray(probabilities, dtype=float)
    gold = np.asarray(gold, dtype=int)
    if (raw.ndim != 2 or not len(raw) or len(raw) != len(gold) or raw.shape[1] < 2
            or not np.isfinite(raw).all() or (raw < 0).any() or (raw > 1).any()
            or ((gold < 0) | (gold >= raw.shape[1])).any()):
        raise ValueError("complete calibration probability matrix required")
    width = raw.shape[1]
    winner, confidence = raw.argmax(axis=1), raw.max(axis=1)
    support = np.bincount(gold, minlength=width)
    active = support > 0
    returned, correct = np.zeros(width, dtype=int), np.zeros(width, dtype=int)
    order = np.argsort(-confidence, kind="stable")
    ranked = confidence[order]
    ends = np.r_[np.flatnonzero(ranked[1:] != ranked[:-1]) + 1, len(order)]
    best = (-1., -1.)
    start = 0
    for end in ends:
        group = order[start:end]
        np.add.at(returned, winner[group], 1)
        hits = group[winner[group] == gold[group]]
        np.add.at(correct, winner[hits], 1)
        score = float((2 * correct[active] / (support[active] + returned[active])).mean())
        best = max(best, (score, float(ranked[start])))
        start = end
    # The all-rejected decision is attainable at 1 only if no recorded score is 1.
    if ranked[0] < 1:
        best = max(best, (0., 1.))
    return dict(threshold=best[1], macro_f1=best[0], queries=len(raw), supported_classes=int(active.sum()))


def routing(confidence, hit, threshold):
    accepted = confidence >= threshold
    errors = accepted & ~hit
    rejected_correct = ~accepted & hit
    return dict(threshold=threshold, accepted=int(accepted.sum()), accepted_errors=int(errors.sum()),
                rejected_correct=int(rejected_correct.sum()), top1_correct=int(hit.sum()),
                coverage=float(accepted.mean()),
                accepted_error_rate=float(errors.sum() / accepted.sum()) if accepted.any() else None,
                correct_top1_rejection_rate=float(rejected_correct.sum() / hit.sum()) if hit.any() else None)


def collect(spec, parent):
    from tools.reports.evaluate_tpo_endpoints import collect_predictions

    roster, assets, ids, _ = source_panel(spec["calibration"], parent)
    gold = [ids.index(row["intent"]) for row in roster]
    results = {}
    reused = spec.get("reused_calibration_predictions", {})
    for arm, expected in spec["calibration"]["policy_ids"].items():
        shards = reused[arm] if arm in reused else spec["shards"]
        records, policy = collect_predictions(shards, arm + ".json", assets)
        if policy != expected:
            raise ValueError("calibration policy differs")
        results[arm] = dict(policy_id=policy, **thresholds([records[row["id"]]["prob"] for row in assets], gold))
    receipt = dict(schema="jev-source-train-harness-thresholds-v1",
                   roster_sha256=spec["calibration"]["roster"]["sha256"], arms=results)
    commit_json(Path(spec["output"]), receipt)
    return receipt

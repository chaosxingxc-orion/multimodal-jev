"""Score paired raw candidate probabilities against a frozen audio roster."""
import math
from pathlib import Path

import numpy as np

from multimodal_jev.paired_event_decisions import compare
from tools.reports.run_native_seal_endpoint import bound


def score_panel(cache, jobs, predictions, protocol, *, thresholds=None):
    root = Path(cache["seal"]["root"])
    seal = bound(dict(path=root / "MANIFEST.json", sha256=cache["seal"]["sha256"]))
    dataset = cache["dataset"]
    rel = f"scoring/{dataset}/targets.json"
    targets = bound(dict(path=root / rel, sha256=seal["files"][rel]))
    ids = [q.id for q in jobs[0][2].questions]
    if (seal["target_tuning_allowed"] is not False or len(jobs) != seal["datasets"][dataset]["recordings"]
            or len(ids) != seal["datasets"][dataset]["candidates"]):
        raise ValueError("complete sealed scoring panel required")
    for (name, row, request), target in zip(jobs, targets, strict=True):
        if (name != dataset or any(row[k] != target[k] for k in ("id", "audio_sha256"))
                or [q.id for q in request.questions] != ids or target["gold"] not in ids):
            raise ValueError("scoring identity differs")
    gold = np.array([ids.index(row["gold"]) for row in targets])
    rows = {}
    values = {}
    winners = {}
    for arm, (records, policy) in predictions.items():
        if set(records) != {row["id"] for row in targets}:
            raise ValueError("prediction population differs from the seal")
        probs = np.array([records[t["id"]]["prob"] for t in targets], dtype=float)
        if probs.shape != (len(targets), len(ids)) or not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
            raise ValueError("complete finite probability matrix required")
        winner, confidence = probs.argmax(axis=1), probs.max(axis=1)
        actual = np.arange(len(ids)) == gold[:, None]
        hit = winner == gold
        nll = -np.log(np.maximum(np.where(actual, probs, 1-probs), np.finfo(float).tiny))
        brier = (probs-actual)**2
        sample = np.arange(len(targets))
        wrong = np.where(actual, -1, probs).max(axis=1)
        bins = []
        bounds = protocol["confidence_bins"]
        if bounds != sorted(set(bounds)) or bounds[0] != 0 or bounds[-1] != 1:
            raise ValueError("frozen confidence bounds required")
        for index, (lower, upper) in enumerate(zip(bounds, bounds[1:])):
            mask = (confidence >= lower) & ((confidence <= upper) if index == len(bounds)-2 else (confidence < upper))
            bins.append(dict(lower=lower, upper=upper, recordings=int(mask.sum()),
                             accuracy=float(hit[mask].mean()) if mask.any() else None))
        metric = dict(policy_id=policy, queries=len(targets), top1_accuracy=float(hit.mean()),
                      mean_top1_probability=float(confidence.mean()), candidate_nll=float(nll.mean()),
                      positive_nll=float(nll[actual].mean()), negative_nll=float(nll[~actual].mean()),
                      candidate_brier=float(brier.mean()), positive_brier=float(brier[actual].mean()),
                      negative_brier=float(brier[~actual].mean()),
                      mean_gold_minus_strongest_wrong=float((probs[sample, gold]-wrong).mean()), confidence_bins=bins)
        if protocol["primary"] == "top1_macro_f1":
            support = np.bincount(gold, minlength=len(ids))
            returned = np.bincount(winner, minlength=len(ids))
            correct = np.bincount(gold[hit], minlength=len(ids))
            metric["top1_macro_f1"] = math.fsum(2*a/b if b else 0 for a,b in zip(correct, support+returned))/len(ids)
        if thresholds is not None:
            from tools.reports.calibrate_tpo_probabilities import routing
            locked = thresholds[arm]
            if locked["policy_id"] != policy or not 0 <= locked["threshold"] <= 1:
                raise ValueError("frozen threshold does not belong to this policy")
            metric["harness"] = routing(confidence, hit, locked["threshold"])
        rows[arm] = metric
        winners[arm] = hit
        values[arm] = {t["id"]: dict(zip(ids, p.tolist(), strict=True))
                       for t,p in zip(targets, probs, strict=True)}
    paired = {}
    for arm in protocol["comparisons"]:
        left, right = winners[arm], winners["tpo"]
        result = dict(fixed=int((~left & right).sum()), regressed=int((left & ~right).sum()))
        if protocol["primary"] == "top1_macro_f1":
            proof = compare(ids, targets, values[arm], values["tpo"], interval=protocol["interval"],
                            cluster=protocol["cluster"])
            result.update(delta=proof["macro_f1_delta"], interval=proof["macro_f1_delta_ci95"])
        else:
            diff = right.astype(int)-left.astype(int)
            boot = protocol["interval"]
            if protocol["cluster"] != "id" or boot["method"] != "paired_cluster_percentile_bootstrap" or boot["confidence"] != .95:
                raise ValueError("paired per-question accuracy interval required")
            rng = np.random.default_rng(boot["seed"])
            samples = rng.choice(diff, (boot["samples"], len(diff))).mean(axis=1)
            result.update(delta=float(diff.mean()), interval=np.quantile(samples,[.025,.975]).tolist())
        paired[arm] = result
    return dict(arms=rows, paired=paired)

"""CPU diagnostics for frozen Bernoulli training targets and native evaluation panels."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def read(path, pin=None):
    content = path.read_bytes()
    if pin and sha256(content).hexdigest() != pin:
        raise ValueError(f"content identity differs: {path}")
    return json.loads(content)


def gap(matrix, gold):
    pos = matrix[np.arange(len(gold)), gold]
    neg = np.where(np.arange(matrix.shape[1]) == gold[:, None], -np.inf, matrix).max(1)
    return pos - neg, pos, neg


def rank_summary(p, y, subset):
    prob = p[np.arange(len(y)), y]
    rank = 1 + (p > prob[:, None]).sum(1)
    return dict(n=int(subset.sum()), tie_first=int(((rank == 1) & subset).sum()),
                second=int(((rank == 2) & subset).sum()),
                third_to_fifth=int(((rank >= 3) & (rank <= 5) & subset).sum()),
                below_fifth=int(((rank > 5) & subset).sum()))


def projection_bounds(probs, gold, bad, reward):
    # Analytic read-only bound for the 206 contract; no target batch, fitting or update.
    if reward["temperature"] != 1 or reward["max_displacement"] <= 0:
        raise ValueError("bound requires unit temperature and positive displacement cap")
    eps, cap = reward["probability_epsilon"], reward["max_displacement"]
    p = np.clip(probs, eps, 1-eps)
    y = (np.arange(p.shape[1]) == gold[:, None]).astype(float)
    z = np.log(p) - np.log1p(-p)
    row = y/p - 1/p.sum(1)[:, None]
    fixed = z + reward["audio_to_label"] * np.clip(row, -cap, cap) + reward["brier"] * np.clip(2*(y-p), -cap, cap)
    # Column utility is nonnegative for true members and nonpositive otherwise.
    lower = gap(fixed, gold)[0]
    upper = lower + 2 * reward["label_to_audio"] * cap
    return dict(guaranteed_strict_first=int((bad & (lower > 0)).sum()),
                conditional=int((bad & (lower <= 0) & (upper > 0)).sum()),
                cannot_be_strict_first=int((bad & (upper <= 0)).sum()))


def recording_evidence(ds, truth, p, gold, ids, wrong, slurp_metadata, minds_metadata):
    groups = defaultdict(list)
    hashes = defaultdict(set)
    for i, r in enumerate(truth):
        groups[r["group"]].append(i)
        hashes[r["audio_sha256"]].add(r["gold"])
    result = dict(wrong_with_cross_gold_audio_hash=sum(len(hashes[t["audio_sha256"]]) > 1
                  for t, bad in zip(truth, wrong) if bad))
    if ds == "MINDS_native14" and minds_metadata:
        import pyarrow.parquet as pq
        annotated = {}
        for source in sorted(minds_metadata.glob("*/*.parquet")):
            columns = ["path", "transcription", "english_transcription"]
            for row in pq.read_table(source, columns=columns).to_pylist():
                uid = "minds14:" + sha256((source.parent.name + chr(0) + row["path"]).encode()).hexdigest()
                if uid in annotated:
                    raise ValueError("duplicate annotation")
                annotated[uid] = (source.parent.name, row)
        if set(annotated) != {r["id"] for r in truth}:
            raise ValueError("MINDS annotation roster differs")
        labels = defaultdict(set)
        keys = []
        for r in truth:
            language, row = annotated[r["id"]]
            if r["group"] != language:
                raise ValueError("MINDS locale differs")
            key = (language, " ".join((row["transcription"] or "").lower().split()))
            keys.append(key)
            if key[1]:
                labels[key].add(r["gold"])
        result["missing_native_text_wrong"] = sum(not k[1] for k,b in zip(keys,wrong) if b)
        result["same_locale_text_cross_gold_wrong"] = sum(len(labels[k]) > 1 for k,b in zip(keys,wrong) if b)
        result["en_US_median_wrong_in_largest_confusion"] = None
        errors = [i for i,t in enumerate(truth) if wrong[i] and t["group"] == "en-US"]
        pair, _ = Counter((ids[gold[i]], ids[p[i].argmax()]) for i in errors).most_common(1)[0]
        pool = sorted((1+int((p[i] > p[i,gold[i]]).sum()),truth[i]["id"]) for i in errors
                      if (ids[gold[i]],ids[p[i].argmax()]) == pair)
        rank, uid = pool[(len(pool)-1)//2]
        result["en_US_median_wrong_in_largest_confusion"] = dict(pair=pair,rank=rank,n=len(pool),
            id=uid, source_text=annotated[uid][1]["transcription"])
    if slurp_metadata is None or ds != "SLURP_clean_TEST60":
        return result
    content = slurp_metadata.read_bytes()
    annotations = {f"slurp:{r['file']}": (row["sentence"], row["slurp_id"], r)
                   for line in content.splitlines() for row in [json.loads(line)] for r in row["recordings"]}
    examples = {}
    siblings = Counter()
    quality = Counter()
    sentence_labels = defaultdict(set)
    for row in truth:
        sentence, source_id, _ = annotations[row["id"]]
        if row["group"] != f"slurp:{source_id}":
            raise ValueError("annotation group mismatch")
        sentence_labels[" ".join(sentence.lower().split())].add(row["gold"])
    for group, indices in groups.items():
        for i in indices:
            if not wrong[i]:
                continue
            sentence, _, rec = annotations[truth[i]["id"]]
            peers = [j for j in indices if j != i and gold[j] == gold[i]
                     and truth[j]["audio_sha256"] != truth[i]["audio_sha256"]]
            peer_correct = any(not wrong[j] for j in peers)
            siblings["sibling_correct" if peer_correct else "siblings_all_wrong" if peers else "no_peer"] += 1
            collision = len(sentence_labels[" ".join(sentence.lower().split())]) > 1
            quality["shared_text_different_gold"] += int(collision)
            quality["source_recording_correct_zero_wer"] += int(rec.get("status") == "correct" and rec.get("wer") == 0)
            predicted = int(p[i].argmax())
            rank = 1 + int((p[i] > p[i, gold[i]]).sum())
            key = "correct_peer" if peer_correct else "all_peers_wrong" if peers else "no_peer"
            # A repeatable illustration near the centre of each subgroup.
            examples.setdefault(key, []).append((rank, truth[i]["id"], sentence, ids[gold[i]],
                                                 ids[predicted], float(p[i,gold[i]]), float(p[i,predicted])))
    result.update(siblings=dict(siblings), quality=dict(quality), annotations_sha256=sha256(content).hexdigest(),
                  middle_by_rank={k: sorted(v)[(len(v)-1)//2] for k,v in examples.items()})
    return result


def train(args):
    import torch
    root = args.targets
    m = read(root / "MANIFEST.json", args.pin)
    old = read(args.reference / "MANIFEST.json", m["bidirectional_contrast"]["baseline_sha256"])
    refs = read(args.reference / "native-probs.json", old["files"]["native-probs.json"])
    rows = read(root / "rows.json", m["files"]["rows.json"])
    uids = sorted(rows, key=lambda uid: rows[uid]["index"])
    if set(uids) != set(refs) or [rows[u]["index"] for u in uids] != list(range(len(rows))):
        raise ValueError("reference roster differs")
    gold = np.array([rows[u]["gold"] for u in uids])
    ref = np.array([refs[u] for u in uids])
    local = root / "targets.pt"
    if sha256(local.read_bytes()).hexdigest() != m["files"]["targets.pt"]:
        raise ValueError("target identity differs")
    target = torch.load(local, weights_only=True, map_location="cpu").numpy()
    if target.shape != (m["schedule"]["epochs"], *ref.shape) or not np.isfinite(target).all():
        raise ValueError("invalid target logits")
    before = gap(ref, gold)[0]
    gaps = np.stack([gap(t, gold)[0] for t in target])
    incorrect = before < 0
    q = 1 / (1 + np.exp(-target.astype(float)))
    ix = np.arange(len(gold))
    neg_idx = np.where(np.arange(ref.shape[1]) == gold[:, None], -np.inf, ref).argmax(1)
    # KL(q || sigmoid(z)), averaged over candidates, at the Stage1 logits.
    descent = (q - ref) / ref.shape[1]
    pos_push = descent[:, ix, gold]
    neg_push = -descent[:, ix, neg_idx]
    result = {}
    for name, mask in {"both_correct_targets": incorrect & (gaps > 0).all(0),
                       "one_correct_target": incorrect & (gaps > 0).any(0) & ~(gaps > 0).all(0),
                       "neither_correct_target": incorrect & (gaps <= 0).all(0)}.items():
        pair = pos_push[:, mask] + neg_push[:, mask]
        result[name] = dict(n=int(mask.sum()), both_directions_each_epoch=int(
            (mask & (pos_push > 0).all(0) & (neg_push > 0).all(0)).sum()),
            median_gold_push=float(np.median(pos_push[:, mask])),
            median_competitor_push=float(np.median(neg_push[:, mask])),
            median_pair_push=float(np.median(pair)), rank=rank_summary(ref, gold, mask),
            groups=len({rows[uids[i]]["group"] for i in np.flatnonzero(mask)}))
    bounds = projection_bounds(ref, gold, incorrect, m["bidirectional_contrast"]["reward"])
    if (bounds["guaranteed_strict_first"] > int((incorrect & (gaps > 0).all(0)).sum())
            or bounds["cannot_be_strict_first"] > int((incorrect & (gaps <= 0).all(0)).sum())):
        raise ValueError("analytic bound incompatible with frozen targets")
    return dict(bounds=bounds, signal_at_initial_actor=result, recordings=len(rows), strict_wrong=int(incorrect.sum()), tied=int((before == 0).sum()),
                fixed_by_epoch=[int((incorrect & (g > 0)).sum()) for g in gaps],
                never_strictly_fixed=int((incorrect & (gaps <= 0).all(0)).sum()),
                correct_lost=[int(((before > 0) & (g <= 0)).sum()) for g in gaps],
                median_ref_gold_never_fixed=float(np.median(gap(ref, gold)[1][incorrect & (gaps <= 0).all(0)])),
                audit=m["bidirectional_contrast"]["audit"])


def panel(args):
    root = args.evaluation
    specs = read(root / "entry/score.json", args.config_pin)["macros"]["NATIVE_SEAL_CONTAINER_CONFIG"]
    receipt = read(root / "metrics.json", args.pin)
    out = {}
    reward = json.loads(args.reward)
    for ds in args.datasets:
        scores = receipt["datasets"][ds]
        cache = specs["endpoints"][ds]["cache"]["seal"]
        seal = Path(cache["root"])
        manifest = read(seal / "MANIFEST.json", receipt["seals"][ds])
        def sealed(name):
            return read(seal / name, manifest["files"][name])
        ids = sealed(f"inference/{ds}/panel.json")["candidate_ids"]
        truth = sealed(f"scoring/{ds}/targets.json")
        gold = np.array([ids.index(row["gold"]) for row in truth])
        roster = [row["id"] for row in truth]
        if len(set(roster)) != len(gold) or len(set(ids)) != len(ids):
            raise ValueError("duplicate roster")
        data = {}
        for arm, score in scores["arms"].items():
            reuse = specs["reused_predictions"].get(ds, {}).get(arm)
            shards = reuse["shards"] if reuse else [
                dict(path=root / Path(s).name / f"{arm}.json") for s in specs["shards"][ds]]
            lookup = {}
            for shard in shards:
                obj = read(Path(shard["path"]), shard.get("sha256"))
                if obj["policy_id"] != score["policy_id"]:
                    raise ValueError("policy differs")
                for row in obj["records"]:
                    if row["id"] in lookup:
                        raise ValueError("duplicate prediction")
                    lookup[row["id"]] = row["prob"]
            if set(roster) != set(lookup):
                raise ValueError("different population")
            p = np.array([lookup[u] for u in roster])
            if p.shape != (len(gold), len(ids)) or not np.all((p >= 0) & (p <= 1)):
                raise ValueError("invalid probability matrix")
            g, pos, neg = gap(p, gold)
            win = p.argmax(1)
            measured = [(win == gold).mean(), p.max(1).mean(), ((1-pos)**2).mean(), g.mean()]
            expected = [score[k] for k in ["top1_accuracy", "mean_top1_probability", "positive_brier",
                                           "mean_gold_minus_strongest_wrong"]]
            if not np.allclose(measured, expected, rtol=0, atol=1e-12):
                raise ValueError("score differs")
            data[arm] = (win, pos, neg, g)
            if arm == "stage1":
                prior_probs = p
            if arm == "tpo":
                tpo_probs = p
        win, pos, neg, g = data["tpo"]
        def confusions(mask):
            pairs = Counter((ids[y], ids[w]) for y, w in zip(gold[mask], win[mask]))
            return sorted(pairs.items(), key=lambda x: (-x[1], x[0]))[:5]
        bad = win != gold
        result = dict(wrong=int(bad.sum()), confusions=confusions(bad), ranks=rank_summary(tpo_probs, gold, bad),
                      recording_evidence=recording_evidence(ds, truth, tpo_probs, gold, ids, bad, args.slurp_metadata, args.minds_metadata))

        result["reference_bounds_on_206_wrong"] = projection_bounds(prior_probs, gold, bad, reward)
        result["endpoint_as_reference_hypothetical"] = projection_bounds(tpo_probs, gold, bad, reward)
        for arm, paired in scores["paired"].items():
            prior, pp, pn, prev_gap = data[arm]
            reg = bad & (prior == gold)
            fix = ~bad & (prior != gold)
            if (int(reg.sum()), int(fix.sum())) != (paired["regressed"], paired["fixed"]):
                raise ValueError("paired counts differ")
            result[arm] = dict(ranks=rank_summary(tpo_probs, gold, reg), fixed=int(fix.sum()), regressed=int(reg.sum()),
                still_wrong=int((bad & (prior != gold)).sum()), top_confusions=confusions(reg),
                gold_down=int((reg & (pos < pp)).sum()), strongest_wrong_up=int((reg & (neg > pn)).sum()),
                both=int((reg & (pos < pp) & (neg > pn)).sum()), tied=int((reg & (g == 0)).sum()))
        # Descriptive risk at a shared coverage. Uniform tie interpolation
        # prevents gold labels or input order from breaking score ties.
        budget = scores["arms"]["tpo"]["harness"]["accepted"]
        risk = {}
        for arm, (w, ppos, pneg, _) in data.items():
            conf = np.maximum(ppos, pneg)
            cutoff = np.sort(conf)[-budget]
            above, tied = conf > cutoff, conf == cutoff
            weight = (budget - int(above.sum())) / int(tied.sum())
            risk[arm] = float(((above & (w != gold)).sum() + weight * (tied & (w != gold)).sum()) / budget)
        result["at_206_coverage"] = dict(accepted=budget, descriptive_error=risk)
        out[ds] = result
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(required=True)
    t = sub.add_parser("train")
    t.add_argument("targets", type=Path)
    t.add_argument("reference", type=Path)
    t.add_argument("pin")
    t.set_defaults(run=train)
    e = sub.add_parser("panel")
    e.add_argument("--minds-metadata", type=Path)
    e.add_argument("--slurp-metadata", type=Path)
    e.add_argument("--reward", required=True, help="frozen contrast reward JSON")
    e.add_argument("evaluation", type=Path)
    e.add_argument("datasets", nargs="+", help="native fixed panels with audited harness output")
    e.add_argument("config_pin")
    e.add_argument("pin")
    e.set_defaults(run=panel)
    a = p.parse_args()
    print(json.dumps(a.run(a), indent=2, allow_nan=False))

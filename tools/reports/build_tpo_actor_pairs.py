"""Freeze source TRAIN review candidates and compile targets after independent review."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import math
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
import unicodedata

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_reference_dataset import digest, validate_record
from multimodal_jev.context_tpo_pair_margin import compile_pairs, validate_spec
from multimodal_jev.local_checkpoint import commit_file, commit_json, sync_directory
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.tpo_view import schedule
from tools.reports.build_tpo_bidirectional_targets import compile_schedule
from tools.reports.build_tpo_view import read, verified

PACKET = "jev-source-audio-pair-review-v1"
REVIEW = "jev-source-audio-pair-submission-v1"


def pinned(root, manifest, name):
    return verified(root / name, manifest["files"][name])


def inputs(config):
    previous, release, reference = (Path(config[key]) for key in ("previous", "release", "reference"))
    old = read(verified(previous / "MANIFEST.json", config["previous_sha256"]))
    parent = read(verified(Path(config["parent_config"]), config["parent_sha256"]))
    source = read(verified(release / "MANIFEST.json", old["release_sha256"]))
    ref = read(verified(reference / "MANIFEST.json", old["reference_sha256"]))
    identity = ref["identity"]
    task = read(pinned(release, source, "tasks.json"))["intent"]
    candidates = [x["id"] for x in task["schema"]["candidates"]]
    if (old["schema"] != "jev-tpo-reward-view-v1" or old["candidate_ids"] != candidates
            or identity["candidate_ids"] != candidates or old["source_policy"] != identity["policy_id"]
            or old["source_policy"] != parent["source_policy_id"]
            or old["release_sha256"] != identity["release_sha256"]
            or old["release_sha256"] != parent["release_sha256"]
            or old["reference_sha256"] != parent["reference_dataset"]["manifest_sha256"]):
        raise ValueError("207 requires the same clean TRAIN, candidates and Stage1 as 206")
    population = {}
    with pinned(release, source, "learning.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if (row["official_split"] != "train" or row["role"] != "learning"
                    or row["known_pending_source"] is not False or row["id"] in population
                    or not row["source_group"] or row["intent"] not in candidates):
                raise ValueError("unique admitted clean source TRAIN required")
            population[row["id"]] = row
    old_rows = read(pinned(previous, old, "rows.json"))
    calibration = read(pinned(previous, old, "calibration.json"))
    calibration_groups = {x["source_group"] for x in calibration}
    calibration_hashes = {x["audio_sha256"] for x in calibration}
    calibration_ids = {x["id"] for x in calibration}
    eligible = {uid: row for uid, row in population.items()
                if uid not in calibration_ids and row["source_group"] not in calibration_groups
                and row["audio_sha256"] not in calibration_hashes}
    pop_keys = read(pinned(reference, ref, "population.json"))
    if set(pop_keys) != set(population) or digest(pop_keys) != identity["population_sha256"]:
        raise ValueError("frozen Stage1 reference population differs")
    records = read(pinned(reference, ref, "records.json"))
    for uid, row in old_rows.items():
        if (uid not in eligible or row["request_key"] != pop_keys[uid]
                or row["audio_sha256"] != eligible[uid]["audio_sha256"]
                or candidates[row["gold"]] != eligible[uid]["intent"]):
            raise ValueError("frozen 206 population binding differs")
    return old, old_rows, calibration, eligible, identity, pop_keys, records, task


def source_text(config, eligible):
    # Source metadata is used for matching only. Humans receive isolated audio.
    by_name = {Path(row["audio"]).name: uid for uid, row in eligible.items()}
    text = defaultdict(set)
    with verified(Path(config["metadata"]), config["metadata_sha256"]).open() as stream:
        for line in stream:
            row = json.loads(line)
            words = tuple(re.findall(r"\w+", unicodedata.normalize("NFKC", row["sentence"]).casefold()))
            for recording in row["recordings"]:
                uid = by_name.get(recording["file"])
                if uid is not None and row["intent"] == eligible[uid]["intent"]:
                    text[eligible[uid]["source_group"]].add(words)
    return {group: next(iter(options)) for group, options in text.items()
            if len(options) == 1 and next(iter(options))}


def templates(words, limit, min_shared):
    for start in range(len(words) + 1):
        for size in range(min(limit, len(words) - start, len(words) - min_shared) + 1):
            yield (words[:start], words[start + size:]), size


def rank_sources(config, eligible, old_rows, records, keys, identity):
    prior = torch.load(pinned(Path(config["previous"]), read(Path(config["previous"]) / "MANIFEST.json"),
                              "targets.pt"), map_location="cpu", weights_only=True)
    label_index = {label: i for i, label in enumerate(identity["candidate_ids"])}
    data = torch.tensor([records[keys[uid]]["prob"] for uid in eligible])
    gold = torch.tensor([label_index[row["intent"]] for row in eligible.values()])
    gold_prob = data.gather(1, gold[:, None])
    data.scatter_(1, gold[:, None], -1.)
    score, rival = data.max(1)
    gaps = (gold_prob.flatten() - score).tolist()
    ranked = {uid: (gold_id, rival_id, gap) for uid, gold_id, rival_id, gap in
              zip(eligible, gold.tolist(), rival.tolist(), gaps, strict=True)}
    old_hashes = {row["audio_sha256"] for row in old_rows.values()}
    fresh = {uid for uid, row in eligible.items() if uid not in old_rows and row["audio_sha256"] not in old_hashes}
    stable, persistent = set(), set()
    for uid, row in old_rows.items():
        gold_id, _, gap = ranked[uid]
        logits = prior[:, row["index"]]
        others = logits.clone()
        others[:, gold_id] = -torch.inf
        margins = logits[:, gold_id] - others.amax(-1)
        if gap > 0 and (margins > 0).all():
            stable.add(uid)
        if gap < 0 and (margins <= 0).all():
            persistent.add(uid)
    return ranked, fresh, stable, persistent


def select_candidates(config, eligible, text, ranked, fresh, stable, persistent, labels):
    rules = config["review"]
    cap = rules["max_recordings_per_group"]
    seed = config["schedule"]["seed"]
    groups = defaultdict(list)
    for uid, row in eligible.items():
        groups[row["source_group"]].append(uid)
    by_label = defaultdict(list)
    for group, uids in groups.items():
        intents = {eligible[uid]["intent"] for uid in uids}
        if len(intents) == 1 and group in text:
            by_label[next(iter(intents))].append(group)
    used_groups, used_audio = set(), set()
    pairs, audit = [], {}
    boundaries = list(rules["boundaries"])
    seen_edges = {tuple(sorted((a, b))) for a, b, _ in boundaries}
    counts = Counter(tuple(sorted((labels[gold], labels[rival])))
                     for gold, rival, gap in ranked.values() if gap < 0)
    for (a, b), _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if len(boundaries) >= rules.get("max_boundaries", len(boundaries)):
            break
        if (a, b) not in seen_edges:
            boundaries.append((a, b, f"Check that the utterances differ only at the {a} / {b} intent boundary."))
            seen_edges.add((a, b))

    def side(group, rival):
        selected, takes, hashes = [], set(), set()
        for uid in sorted(groups[group], key=lambda uid: (
                uid not in fresh, uid not in persistent,
                not (ranked[uid][1] == rival and ranked[uid][2] < 0), digest([seed, uid]))):
            # Same acquisition with two channels is one take.
            take = Path(eligible[uid]["audio"]).stem.removesuffix("-headset")
            sha = eligible[uid]["audio_sha256"]
            if take in takes or sha in hashes or sha in used_audio:
                continue
            selected.append(uid)
            takes.add(take)
            hashes.add(sha)
            if len(selected) == cap:
                break
        return selected, hashes

    for left, right, condition in boundaries:
        index = defaultdict(list)
        for group in by_label[left]:
            if group not in used_groups:
                for template, change in templates(text[group], rules["max_changed_words"], rules["min_shared_words"]):
                    index[template].append((group, change))
        matches = {}
        for group in by_label[right]:
            if group in used_groups:
                continue
            for template, change in templates(text[group], rules["max_changed_words"], rules["min_shared_words"]):
                for other, other_change in index.get(template, ()):
                    if text[group] == text[other]:
                        continue
                    key = (other, group)
                    cost = (max(change, other_change), -sum(map(len, template)))
                    matches[key] = min(cost, matches.get(key, cost))
        taken = 0
        for a, b in sorted(matches, key=lambda pair: (
                not any(uid in persistent for group in pair for uid in groups[group]),
                matches[pair], digest([seed, pair]))):
            if a in used_groups or b in used_groups:
                continue
            a_ids, a_sha = side(a, labels.index(right))
            b_ids, b_sha = side(b, labels.index(left))
            if not a_ids or not b_ids or a_sha & b_sha or not ({a_ids[0], b_ids[0]} & fresh):
                continue
            pair_id = digest([seed, a, b])
            pairs.append(dict(pair_id=pair_id, sides=[a_ids, b_ids], boundary=[left, right], condition=condition))
            used_audio.update(a_sha | b_sha)
            used_groups.update((a, b))
            taken += 1
            if taken == rules["max_pairs_per_boundary"]:
                break
        audit[f"{left}:{right}"] = dict(near_matches=len(matches), nominated=taken)
    # Each independent hard recording needs the same blinded admission as a
    # pair member. Reserve groups first so neither it nor replay duplicates a pair.
    hard = []
    for group in sorted(groups, key=lambda g: digest([seed, "hard-group", g])):
        if group in used_groups:
            continue
        takes = set()
        candidates = sorted((uid for uid in groups[group] if ranked[uid][2] < 0),
                            key=lambda uid: (uid not in persistent, ranked[uid][2], digest([seed, uid])))
        for uid in candidates:
            row = eligible[uid]
            take = Path(row["audio"]).stem.removesuffix("-headset")
            if take in takes or row["audio_sha256"] in used_audio:
                continue
            hard.append(uid)
            used_audio.add(row["audio_sha256"])
            used_groups.add(group)
            takes.add(take)
            if len(takes) == rules.get("max_hard_per_group", cap):
                break
    paired_recordings = sum(len(side) for pair in pairs for side in pair["sides"])
    fraction = rules["replay_max_fraction"]
    replay_limit = math.floor((paired_recordings + len(hard)) * fraction / (1 - fraction))
    replay = []
    if replay_limit:
        for uid in sorted(stable, key=lambda uid: digest([seed, "replay", uid])):
            row = eligible[uid]
            if row["source_group"] not in used_groups and row["audio_sha256"] not in used_audio:
                replay.append(uid)
                used_groups.add(row["source_group"])
                used_audio.add(row["audio_sha256"])
                if len(replay) == replay_limit:
                    break
    if not pairs or not replay:
        raise ValueError("source TRAIN has no complete candidate pairs or disjoint replay")
    return pairs, hard, replay, audit


def packet_config(config):
    return digest({name: config[name] for name in
                   ("previous_sha256", "parent_sha256", "metadata_sha256", "review", "schedule", "loss")})


def load_packet(config):
    root = Path(config["packet"])
    manifest = read(root / "MANIFEST.json")
    if manifest["schema"] != PACKET or manifest["design_sha256"] != packet_config(config):
        raise ValueError("review packet is not bound to the frozen design")
    pins = [(root / directory / name, sha)
            for directory in ("private", "listening", "relation")
            for name, sha in manifest["files"][directory].items()]
    with ThreadPoolExecutor(max_workers=config["io_workers"]) as pool:
        for _ in pool.map(lambda item: verified(*item), pins):
            pass
    return manifest


def prepare(config):
    output = Path(config["packet"])
    if output.exists():
        return load_packet(config)
    check_paths({"review packet": output})
    old, old_rows, _, eligible, identity, keys, refs, task = inputs(config)
    ranked, fresh, stable, persistent = rank_sources(config, eligible, old_rows, refs, keys, identity)
    pairs, hard, replay, audit = select_candidates(
        config, eligible, source_text(config, eligible), ranked, fresh, stable, persistent, old["candidate_ids"])
    pair_members = {uid for pair in pairs for side in pair["sides"] for uid in side}
    review_ids = {uid: digest([config["schedule"]["seed"], "blind-listening", eligible[uid]["audio_sha256"]])
                  for uid in sorted(pair_members | set(hard))}
    assets = {rid: eligible[uid] for uid, rid in review_ids.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".review-", dir=output.parent) as staging:
        root = Path(staging) / "packet"
        for directory in ("private", "listening/audio", "relation"):
            (root / directory).mkdir(parents=True, mode=0o700 if directory == "private" else 0o755)

        def copy_asset(item):
            review_id, source = item
            dest = root / "listening/audio" / f"{review_id}.flac"
            shutil.copyfile(source["audio"], dest)
            if file_hash(dest) != source["audio_sha256"]:
                raise ValueError("source audio content changed")
            return f"audio/{dest.name}", source["audio_sha256"]

        with ThreadPoolExecutor(max_workers=config["io_workers"]) as pool:
            audio_pins = dict(pool.map(copy_asset, sorted(assets.items())))
        roster = [{"review_id": rid, "audio": f"audio/{rid}.flac"} for rid in sorted(assets)]
        commit_json(root / "listening/candidates.json", task["schema"]["candidates"])
        commit_json(root / "listening/roster.json", roster)
        relations = [{"pair_id": pair["pair_id"], "audio": [
            f"../listening/audio/{review_ids[side[0]]}.flac" for side in pair["sides"]],
            "condition": pair["condition"]} for pair in pairs]
        commit_json(root / "relation/roster.json", relations)
        commit_json(root / "private/bindings.json", dict(pairs=pairs, hard=hard, replay=replay, review_ids=review_ids))
        manifest = dict(schema=PACKET, design_sha256=packet_config(config),
                        review_state="awaiting_human_reviews", eligible_train=len(eligible),
                        possible_new_recordings=len(fresh), candidate_pairs=len(pairs),
                        review_recordings=len(assets), pair_group_recordings=len(pair_members),
                        hard_single_candidates=len(hard), fresh_pair_recordings=len(pair_members & fresh),
                        unique_source_groups=len({eligible[u]["source_group"] for u in pair_members | set(hard) | set(replay)}),
                        candidate_max_recordings=len(assets) + len(replay), replay_candidates=len(replay),
                        reference_covered=len(assets) + len(replay), boundary_counts=audit,
                        files={name: {str(p.relative_to(root / name)): file_hash(p)
                                      for p in sorted((root / name).glob("*.json"))}
                               for name in ("private", "listening", "relation")})
        manifest["files"]["listening"].update(audio_pins)
        commit_json(root / "MANIFEST.json", manifest)
        for directory, entries in (("listening", {x["review_id"]: dict(status="unreviewed", label=None) for x in roster}),
                                   ("relation", {x["pair_id"]: dict(status="unreviewed", evidence="") for x in relations})):
            template = dict(schema=REVIEW, reviewer_id="", started_at="", finished_at="",
                            packet_sha256=file_hash(root / "MANIFEST.json"), answers=entries)
            if directory == "listening":
                template.update(blind_attested=None, audio_only=None)
            commit_json(root / directory / "submission-template.json", template)
        root.rename(output)
        sync_directory(output.parent)
    return manifest


def load_submissions(config, packet_manifest):
    receipt_path = Path(config["submissions"])
    receipt = read(receipt_path)
    packet_sha = file_hash(Path(config["packet"]) / "MANIFEST.json")
    if (receipt.get("schema") != REVIEW or receipt.get("packet_sha256") != packet_sha
            or set(receipt["files"]) != {"listening-a.json", "listening-b.json", "relation.json"}
            or packet_manifest["review_state"] != "awaiting_human_reviews"):
        raise ValueError("frozen packet and three pinned review submissions required")
    return read_submissions(receipt_path, receipt)


def read_submissions(receipt_path, receipt):
    packet_sha = receipt["packet_sha256"]
    # Read the exact bytes that were pinned by the review coordinator.
    responses = {name: read(verified(receipt_path.parent / name, sha))
                 for name, sha in receipt["files"].items()}
    for response in responses.values():
        if (response.get("schema") != REVIEW or response.get("packet_sha256") != packet_sha
                or not isinstance(response.get("reviewer_id"), str) or not response["reviewer_id"].strip()
                or not isinstance(response.get("answers"), dict)):
            raise ValueError("review identity or packet binding missing")
        started, finished = (datetime.fromisoformat(response[k]) for k in ("started_at", "finished_at"))
        if started.tzinfo is None or finished.tzinfo is None or started >= finished:
            raise ValueError("explicit review period required")
    first, second, relation = (responses[key] for key in ("listening-a.json", "listening-b.json", "relation.json"))
    if first["reviewer_id"].strip().casefold() == second["reviewer_id"].strip().casefold() or any(
            r.get("blind_attested") is not True or r.get("audio_only") is not True for r in (first, second)):
        raise ValueError("two distinct independent audio-only review attestations required")
    if any(relation["reviewer_id"].strip().casefold() == r["reviewer_id"].strip().casefold()
           and datetime.fromisoformat(relation["started_at"]) < datetime.fromisoformat(r["finished_at"])
           for r in (first, second)):
        raise ValueError("pair-aware review must follow the reviewer's blind pass")
    return first["answers"], second["answers"], relation["answers"]


def seal_submissions(config):
    packet = load_packet(config)
    path = Path(config["submissions"])
    if path.exists():
        load_submissions(config, packet)
        return read(path)
    names = ("listening-a.json", "listening-b.json", "relation.json")
    receipt = dict(schema=REVIEW, packet_sha256=file_hash(Path(config["packet"]) / "MANIFEST.json"),
                   files={name: file_hash(path.parent / name) for name in names})
    answers = read_submissions(path, receipt)
    old, old_rows, _, eligible, *_ = inputs(config)
    bindings = read(Path(config["packet"]) / "private/bindings.json")
    previous = {row["audio_sha256"] for row in old_rows.values()}
    fresh = {uid for uid, row in eligible.items() if uid not in old_rows and row["audio_sha256"] not in previous}
    admit(bindings, answers, eligible, old["candidate_ids"], fresh)
    commit_json(path, receipt)
    return receipt


def admit(bindings, answers, eligible, labels, fresh):
    first, second, relation = answers
    review_ids = bindings["review_ids"]
    if set(first) != set(second) or set(first) != set(review_ids.values()) or set(relation) != {
            pair["pair_id"] for pair in bindings["pairs"]}:
        raise ValueError("review coordinates differ from the frozen roster")
    for review in (first, second):
        for item in review.values():
            if ((item["status"] == "unique" and item["label"] not in labels)
                    or (item["status"] in ("ambiguous", "unintelligible") and item["label"] is not None)
                    or item["status"] not in ("unique", "ambiguous", "unintelligible", "unreviewed")
                    or (item["status"] == "unreviewed" and item["label"] is not None)):
                raise ValueError("listening review needs an explicit allowed result")
    rejected = Counter()
    selected, approved, admitted_pairs = {}, set(), []

    def accepted(uid):
        rid = review_ids[uid]
        a, b = first[rid], second[rid]
        if a["status"] != "unique" or b["status"] != "unique":
            rejected["missing_or_nonunique_recording"] += 1
            return False
        if a["label"] != b["label"] or a["label"] != eligible[uid]["intent"]:
            rejected["label_disagreement"] += 1
            return False
        return True

    for pair in bindings["pairs"]:
        pair_id = pair["pair_id"]
        assessment = relation[pair_id]
        if assessment["status"] not in ("confirmed", "rejected", "unreviewed") or (
                assessment["status"] != "unreviewed" and not assessment["evidence"].strip()):
            raise ValueError("pair review needs a decision and evidence")
        sides = [[uid for uid in side if accepted(uid)] for side in pair["sides"]]
        if assessment["status"] != "confirmed" or any(
                not side or side[0] != original[0] for side, original in zip(sides, pair["sides"], strict=True)):
            rejected["pair_unconfirmed_or_missing_side"] += 1
            continue
        for side in sides:
            for i, uid in enumerate(side):
                selected[uid] = dict(pair_id=pair_id, kind="pair" if i == 0 else "rerecording")
                approved.add(uid)
        admitted_pairs.append(dict(pair, sides=sides))
    for uid in bindings["hard"]:
        if accepted(uid):
            selected[uid] = dict(pair_id=None, kind="hard_single")
            approved.add(uid)
    if not any(set(pair["sides"][0][:1] + pair["sides"][1][:1]) & fresh for pair in admitted_pairs):
        raise ValueError("no complete reviewed pair with a primary recording absent from 206")
    return selected, approved, admitted_pairs, rejected


def replay_for(config, bindings, admitted_count):
    fraction = config["review"]["replay_max_fraction"]
    return bindings["replay"][:math.floor(admitted_count * fraction / (1 - fraction))]


def build(config):
    packet = load_packet(config)
    reviews = load_submissions(config, packet)
    old, old_rows, calibration, eligible, identity, keys, refs, _ = inputs(config)
    bindings = read(Path(config["packet"]) / "private/bindings.json")
    ranked, fresh, stable, _ = rank_sources(config, eligible, old_rows, refs, keys, identity)
    selected, approved, pairs, rejected = admit(bindings, reviews, eligible, old["candidate_ids"], fresh)
    replay = replay_for(config, bindings, len(selected))
    if not replay or not set(replay) <= stable:
        raise ValueError("stable historical replay required")
    selected.update({uid: dict(pair_id=None, kind="replay") for uid in replay})
    groups = Counter(eligible[uid]["source_group"] for uid in selected)
    if (max(groups.values()) > config["review"]["max_recordings_per_group"]
            or len({eligible[uid]["audio_sha256"] for uid in selected}) != len(selected)):
        raise ValueError("duplicate recording or excessive source group")
    validate_spec(config["loss"])
    if config["loss"]["weight"] != 1 / len(old["candidate_ids"]):
        raise ValueError("registered pair margin weight must equal one candidate's KL weight")
    order = sorted(selected)
    group_ids = {group: i for i, group in enumerate(sorted(groups))}
    rows, probs = {}, {}
    for index, uid in enumerate(order):
        record = refs[keys[uid]]
        validate_record(record, identity)
        if record["request_id"] != keys[uid]:
            raise ValueError("fixed reference request identity differs")
        gold, rival, gap = ranked[uid]
        source = eligible[uid]
        rows[uid] = dict(**selected[uid], index=index, gold=gold, competitor=rival, ref_gap=gap,
                         group=group_ids[source["source_group"]], audio_sha256=source["audio_sha256"],
                         request_key=keys[uid])
        probs[uid] = record["prob"]
    design = config["schedule"]
    batches = [dict(b, mode="joint") for b in schedule(rows, **design)]
    steps = design["epochs"] * math.ceil(len(rows) / design["global_batch"])
    if len(batches) != steps or steps < 3:
        raise ValueError(f"reviewed view must support three formal updates under the registered schedule: N={len(rows)}")
    rewards = old["bidirectional_contrast"]["reward"]
    targets, batches, real, audit, ordering = compile_schedule(
        dict(schedule=design, reward=rewards, updates=steps), rows, batches, order, probs)
    rival, active = compile_pairs(torch.tensor([probs[uid] for uid in order]), targets,
                                  torch.tensor([rows[uid]["gold"] for uid in order]),
                                  torch.tensor([uid in approved for uid in order]), config["loss"]["logit_margin"])
    counts = [sum(bool(active[b["epoch"], rows[s["id"]]["index"]]) for s in b["slots"] if not s["padding"])
              for b in batches]
    if not active.any(dim=1).all():
        raise ValueError("reviewed view has an epoch without eligible sorting corrections")
    manifest = {name: old[name] for name in (
        "schema", "source_policy", "view_sha256", "reference_sha256", "release_sha256", "candidate_ids", "calibration")}
    manifest.update(schedule=design, recordings=len(rows), updates=steps, real_exposures=sum(real),
                    batch_real=real, reviewed_new_view=True,
                    bidirectional_contrast=dict(reward=rewards, batch_order_sha256=ordering, audit=audit),
                    actor_pair_margin=dict(loss=config["loss"], epochs=[b["epoch"] for b in batches],
                                           batch_eligible=counts, eligible_by_epoch=active.sum(dim=1).tolist()),
                    review=dict(packet_sha256=file_hash(Path(config["packet"]) / "MANIFEST.json"),
                                receipt_sha256=file_hash(Path(config["submissions"])),
                                existing_recordings_new_to_view=len(set(rows) & fresh), newly_recorded=0,
                                new_source_groups=len({eligible[uid]["source_group"] for uid in fresh & set(rows)} -
                                                      {eligible[uid]["source_group"] for uid in old_rows}),
                                complete_pairs=len(pairs), rerecordings=sum(
                                    row["kind"] == "rerecording" for row in rows.values()),
                                hard_singles=sum(row["kind"] == "hard_single" for row in rows.values()),
                                replay=len(replay), rejected=dict(rejected)),
                    scale_targets=dict(recordings=config["review"]["min_train_recordings"],
                                       complete_pairs=config["review"]["min_complete_pairs"],
                                       eligible_per_epoch=config["review"]["min_eligible_exposures_per_epoch"]))
    destination = Path(config["output"])
    check_paths({"207 targets": destination})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pair-targets-", dir=destination.parent) as staging:
        stage = Path(staging) / "targets"
        stage.mkdir()
        for name, value in (("rows", rows), ("batches", batches), ("calibration", calibration), ("pairs", pairs)):
            commit_json(stage / f"{name}.json", value)
        commit_file(stage / "targets.pt", lambda f: torch.save(targets, f))
        commit_file(stage / "actor-pairs.pt", lambda f: torch.save(dict(rival=rival, enabled=active), f))
        manifest["files"] = {path.name: file_hash(path) for path in sorted(stage.iterdir())}
        commit_json(stage / "MANIFEST.json", manifest)
        if destination.exists():
            verified(destination / "MANIFEST.json", file_hash(stage / "MANIFEST.json"))
            for name in manifest["files"]:
                pinned(destination, manifest, name)
        else:
            stage.rename(destination)
            sync_directory(destination.parent)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "seal-reviews", "build"))
    parser.add_argument("config")
    parser.add_argument("local")
    args = parser.parse_args()
    configuration, _ = load_config(args.config, args.local)
    torch.set_num_threads(configuration["cpu_threads"])
    operations = {"prepare": prepare, "seal-reviews": seal_submissions, "build": build}
    result = operations[args.action](configuration)
    print(json.dumps({key: value for key, value in result.items() if key not in ("files", "batch_real")}))

"""Compile source TRAIN audio targets from frozen reference-text evidence."""
import argparse
from collections import Counter, defaultdict
from itertools import zip_longest
import json
import math
from pathlib import Path
import re
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

TEXT_SCREEN = "jev-source-reference-text-screen-v1"


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
    """Bind released reference sentences to eligible recordings, before applying any text rule."""
    by_name = defaultdict(list)
    for uid, row in eligible.items():
        by_name[Path(row["audio"]).name].append(uid)
    evidence = defaultdict(list)
    with verified(Path(config["metadata"]), config["metadata_sha256"]).open() as stream:
        for line in stream:
            row = json.loads(line)
            words = tuple(re.findall(r"\w+", unicodedata.normalize("NFKC", row["sentence"]).casefold()))
            for recording in row["recordings"]:
                for uid in by_name.get(recording["file"], ()):
                    evidence[uid].append((row["intent"], words, recording))
    group_options = defaultdict(set)
    for uid, bindings in evidence.items():
        group_options[eligible[uid]["source_group"]].update((label, words) for label, words, _ in bindings)

    rules = config["text_screen"]
    patterns = {label: re.compile(r"\b(?:" + "|".join(expressions) + r")\b")
                for label, expressions in rules["intent_cues"].items()}
    if not patterns or not set(patterns) <= {row["intent"] for row in eligible.values()}:
        raise ValueError("Text evidence rules must refer to the released intent panel")
    exclude = re.compile(rules["exclude"])
    text, rejected, results = {}, Counter(), {}
    for uid, row in eligible.items():
        bindings = evidence[uid]
        if not bindings:
            reason = "missing_recording_binding"
        elif len(bindings) != 1 or len(by_name[Path(row["audio"]).name]) != 1 or len(
                group_options[row["source_group"]]) != 1 or bindings[0][0] != row["intent"]:
            reason = "conflicting_recording_or_group"
        else:
            _, words, recording = bindings[0]
            wer = recording.get("wer")
            if (recording.get("status") != rules["recording_status"]
                    or type(wer) not in (int, float) or not math.isfinite(wer)
                    or wer < 0):
                reason = "recording_quality_unconfirmed"
            else:
                sentence = " ".join(words)
                if not words or exclude.search(sentence):
                    reason = "negation_or_unresolved_text"
                else:
                    matches = {label: list(pattern.finditer(sentence)) for label, pattern in patterns.items()}
                    found = {label: spans for label, spans in matches.items() if spans}
                    if set(found) == {row["intent"]}:
                        spans = [(sentence.count(" ", 0, m.start()), sentence.count(" ", 0, m.end()) + 1)
                                 for m in found[row["intent"]]]
                        text[uid] = dict(words=words, cue_spans=spans, metadata_wer=wer)
                        reason = "accepted"
                    else:
                        reason = ("no_intent_cue" if not found else "conflicting_intent_cues"
                                  if len(found) > 1 else "intent_cue_disagrees_with_label")
        results[uid] = reason
        if reason != "accepted":
            rejected[reason] += 1
    return text, rejected, results


def cue_changes(left, right, template):
    """Require each side's changed segment to carry evidence for its own intent."""
    prefix, suffix = template
    start = len(prefix)
    ends = [len(row["words"]) - len(suffix) for row in (left, right)]
    left_change, right_change = [row["words"][start:end] for row, end in zip((left, right), ends)]
    if not left_change or not right_change or (left_change[0] == right_change[0]
                                               or left_change[-1] == right_change[-1]):
        return False
    return all(any(a < end and b > start for a, b in row["cue_spans"])
               for row, end in zip((left, right), ends))


def templates(words, limit, min_shared):
    for start in range(len(words) + 1):
        for size in range(min(limit, len(words) - start, len(words) - min_shared) + 1):
            if size:
                yield (words[:start], words[start + size:]), size


def rank_sources(config, eligible, old_rows, records, keys, identity):
    prior = torch.load(pinned(Path(config["previous"]), read(Path(config["previous"]) / "MANIFEST.json"),
                              "targets.pt"), map_location="cpu", weights_only=True)
    label_index = {label: i for i, label in enumerate(identity["candidate_ids"])}
    data = torch.tensor([records[keys[uid]]["prob"] for uid in eligible])
    gold = torch.tensor([label_index[row["intent"]] for row in eligible.values()])
    gold_prob = data.gather(1, gold[:, None])
    gold_ranks = (1 + (data > gold_prob).sum(dim=1)).tolist()
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
    return ranked, fresh, stable, persistent, dict(zip(eligible, gold_ranks, strict=True))


def select_candidates(config, eligible, text, ranked, fresh, stable, persistent, labels):
    rules = config["selection"]
    cap = rules["max_recordings_per_group"]
    seed = config["schedule"]["seed"]
    groups = defaultdict(list)
    for uid in text:
        groups[eligible[uid]["source_group"]].append(uid)
    by_label, group_text = defaultdict(list), {}
    for group, uids in groups.items():
        intents = {eligible[uid]["intent"] for uid in uids}
        if len(intents) == 1:
            by_label[next(iter(intents))].append(group)
            group_text[group] = text[uids[0]]
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
            boundaries.append((a, b, f"Text cues distinguish {a} from {b}"))
            seen_edges.add((a, b))

    def side(group, rival):
        selected, takes, hashes = [], set(), set()
        for uid in sorted(groups[group], key=lambda uid: (
                uid not in fresh, uid not in persistent,
                not (ranked[uid][1] == rival and ranked[uid][2] < 0), text[uid]["metadata_wer"], digest([seed, uid]))):
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
                for template, change in templates(group_text[group]["words"], rules["max_changed_words"], rules["min_shared_words"]):
                    index[template].append((group, change))
        matches = {}
        for group in by_label[right]:
            if group in used_groups:
                continue
            for template, change in templates(group_text[group]["words"], rules["max_changed_words"], rules["min_shared_words"]):
                for other, other_change in index.get(template, ()):
                    a_text, b_text = group_text[other], group_text[group]
                    if a_text["words"] == b_text["words"] or not cue_changes(a_text, b_text, template):
                        continue
                    key = (other, group)
                    cost = (max(change, other_change), -sum(map(len, template)))
                    evidence = (cost, template)
                    matches[key] = min(evidence, matches.get(key, evidence))
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
            _, (prefix, suffix) = matches[a, b]
            changed = [[len(prefix), len(entry["words"]) - len(suffix)]
                       for entry in (group_text[a], group_text[b])]
            pairs.append(dict(pair_id=pair_id, sides=[a_ids, b_ids], boundary=[left, right],
                              condition=condition, changed_spans=changed))
            used_audio.update(a_sha | b_sha)
            used_groups.update((a, b))
            taken += 1
            if taken == rules["max_pairs_per_boundary"]:
                break
        audit[f"{left}:{right}"] = dict(near_matches=len(matches), nominated=taken)
    # Keep pairs, hard singles and fresh coverage disjoint by source group.
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
    # Use one recording per source before taking another recording of that source.
    fresh_groups = defaultdict(list)
    for group, uids in groups.items():
        if group in used_groups:
            continue
        admitted, takes = [], set()
        for uid in sorted((uid for uid in uids if uid in fresh),
                          key=lambda uid: (text[uid]["metadata_wer"], digest([seed, uid]))):
            take = Path(eligible[uid]["audio"]).stem.removesuffix("-headset")
            if take not in takes:
                takes.add(take)
                admitted.append(uid)
                if len(admitted) == cap:
                    break
        if admitted:
            fresh_groups[eligible[admitted[0]]["intent"]].append(admitted)
    coverage = []
    pools = [sorted(fresh_groups[label], key=lambda ids: digest([seed, ids[0]]))
             for label in sorted(fresh_groups)]
    for take in range(cap):
        for level in zip_longest(*pools):
            for uids in level:
                if uids is None or take >= len(uids):
                    continue
                if len(coverage) >= rules["max_fresh_coverage_recordings"]:
                    break
                uid = uids[take]
                row = eligible[uid]
                if row["audio_sha256"] not in used_audio:
                    coverage.append(uid)
                    used_groups.add(row["source_group"])
                    used_audio.add(row["audio_sha256"])
            if len(coverage) >= rules["max_fresh_coverage_recordings"]:
                break
        if len(coverage) >= rules["max_fresh_coverage_recordings"]:
            break
    fraction = rules["replay_max_fraction"]
    replay_limit = math.floor((paired_recordings + len(hard) + len(coverage)) * fraction / (1 - fraction))
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
    return pairs, hard, coverage, replay, audit


def build(config):
    old, old_rows, calibration, eligible, identity, keys, refs, _ = inputs(config)
    ranked, fresh, stable, persistent, gold_ranks = rank_sources(config, eligible, old_rows, refs, keys, identity)
    text, rejected, screen_results = source_text(config, eligible)
    pairs, hard, coverage, replay, boundary_counts = select_candidates(
        config, eligible, text, ranked, fresh, stable, persistent, old["candidate_ids"])
    selected = {}
    for pair in pairs:
        for side in pair["sides"]:
            for index, uid in enumerate(side):
                selected[uid] = dict(pair_id=pair["pair_id"], kind="pair" if index == 0 else "rerecording")
    selected.update({uid: dict(pair_id=None, kind="hard_single") for uid in hard})
    selected.update({uid: dict(pair_id=None, kind="new_coverage") for uid in coverage})
    approved = set(selected)
    selected.update({uid: dict(pair_id=None, kind="replay") for uid in replay})
    groups = Counter(eligible[uid]["source_group"] for uid in selected)
    if (max(groups.values()) > config["selection"]["max_recordings_per_group"]
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
        raise ValueError(f"text-screened view must support three formal updates under the registered schedule: N={len(rows)}")
    rewards = old["bidirectional_contrast"]["reward"]
    targets, batches, real, audit, ordering = compile_schedule(
        dict(schedule=design, reward=rewards, updates=steps), rows, batches, order, probs)
    rival, active = compile_pairs(torch.tensor([probs[uid] for uid in order]), targets,
                                  torch.tensor([rows[uid]["gold"] for uid in order]),
                                  torch.tensor([uid in approved for uid in order]), config["loss"]["logit_margin"])
    counts = [sum(bool(active[b["epoch"], rows[s["id"]]["index"]]) for s in b["slots"] if not s["padding"])
              for b in batches]
    if not active.any(dim=1).all():
        raise ValueError("text-screened view has an epoch without eligible sorting corrections")
    manifest = {name: old[name] for name in (
        "schema", "source_policy", "view_sha256", "reference_sha256", "release_sha256", "candidate_ids", "calibration")}
    current_hard = {uid for uid in eligible if ranked[uid][2] < 0}
    manifest.update(schedule=design, recordings=len(rows), updates=steps, real_exposures=sum(real),
                    batch_real=real, text_screened_new_view=True,
                    bidirectional_contrast=dict(reward=rewards, batch_order_sha256=ordering, audit=audit),
                    actor_pair_margin=dict(loss=config["loss"], epochs=[b["epoch"] for b in batches],
                                           batch_eligible=counts, eligible_by_epoch=active.sum(dim=1).tolist()),
                    text_screen=dict(schema=TEXT_SCREEN, metadata_sha256=config["metadata_sha256"],
                                rules_sha256=digest(config["text_screen"]), eligible_train=len(eligible),
                                accepted_before_sampling=len(text),
                                accepted_strict_misranked=sum(ranked[uid][2] < 0 for uid in text),
                                strict_misranked=len(current_hard),
                                misranked_by_result=dict(Counter(screen_results[uid] for uid in sorted(current_hard))),
                                misranked_gold_rank=dict(Counter(gold_ranks[uid] for uid in sorted(current_hard))),
                                admitted_misranked_gold_rank=dict(Counter(gold_ranks[uid] for uid in sorted(approved)
                                                                         if uid in current_hard)),
                                boundary_counts=boundary_counts, reference_covered=len(rows),
                                selected_distinct_groups=len(groups),
                                selected_with_nonzero_metadata_wer=sum(text[uid]["metadata_wer"] > 0
                                                                       for uid in approved),
                                selected_intents=len({row["gold"] for row in rows.values()}),
                                existing_recordings_new_to_view=len(set(rows) & fresh), newly_recorded=0,
                                new_source_groups=len({eligible[uid]["source_group"] for uid in fresh & set(rows)} -
                                                      {eligible[uid]["source_group"] for uid in old_rows}),
                                complete_pairs=len(pairs), rerecordings=sum(
                                    row["kind"] == "rerecording" for row in rows.values()),
                                hard_singles=sum(row["kind"] == "hard_single" for row in rows.values()),
                                new_coverage=len(coverage), replay=len(replay), rejected=dict(rejected)),
                    scale_targets=dict(recordings=config["selection"]["min_train_recordings"],
                                       complete_pairs=config["selection"]["min_complete_pairs"],
                                       eligible_per_epoch=config["selection"]["min_eligible_exposures_per_epoch"]))
    destination = Path(config["output"])
    check_paths({"207 targets": destination})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pair-targets-", dir=destination.parent) as staging:
        stage = Path(staging) / "targets"
        stage.mkdir()
        screen = {}
        for uid, result in screen_results.items():
            evidence = dict(result=result)
            if uid in approved:
                evidence.update(text_sha256=digest(text[uid]["words"]), cue_spans=text[uid]["cue_spans"],
                                metadata_wer=text[uid]["metadata_wer"], gold_rank=gold_ranks[uid])
            screen[uid] = evidence
        for name, value in (("rows", rows), ("batches", batches), ("calibration", calibration),
                            ("pairs", pairs), ("text-screen", screen)):
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
    parser.add_argument("action", choices=("build",))
    parser.add_argument("config")
    parser.add_argument("local")
    args = parser.parse_args()
    configuration, _ = load_config(args.config, args.local)
    torch.set_num_threads(configuration["cpu_threads"])
    result = build(configuration)
    print(json.dumps({key: value for key, value in result.items() if key not in ("files", "batch_real")}))

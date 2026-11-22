"""Build a fixed projection for every admitted, calibration-disjoint source recording."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
from tempfile import TemporaryDirectory

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_reference_dataset import digest, validate_record
from multimodal_jev.local_checkpoint import commit_file, commit_json, sync_directory
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.build_tpo_actor_pairs import inputs, pinned
from tools.reports.build_tpo_bidirectional_targets import compile_schedule
from tools.reports.build_tpo_view import verified


def shuffled_batches(ids, design):
    size = design["global_batch"]
    batches = []
    for epoch in range(design["epochs"]):
        order = sorted(ids)
        random.Random(design["seed"] + epoch).shuffle(order)
        for offset in range(0, len(order), size):
            block = order[offset:offset+size]
            slots = [dict(id=uid, weight=1., padding=False) for uid in block]
            slots.extend(dict(id=block[0], weight=0., padding=True) for _ in range(size-len(block)))
            batches.append(dict(index=len(batches), epoch=epoch, mode="joint", slots=slots))
    return batches


def build(config):
    old, prior, calibration, eligible, identity, keys, refs, _ = inputs(config)
    labels = old["candidate_ids"]
    loss = config["loss"]
    if not 0 < loss["margin"] < 1 or loss["weight"] != 1/len(labels):
        raise ValueError("invalid probability gap specification")
    rows, probs, excluded = {}, {}, {}
    seen = set()
    hash_labels = defaultdict(set)
    outcomes = defaultdict(list)
    for row in eligible.values():
        hash_labels[row["audio_sha256"]].add(row["intent"])
    for uid, row in sorted(eligible.items()):
        audio = row["audio_sha256"]
        if len(hash_labels[audio]) != 1 or audio in seen:
            excluded[uid] = "conflicting_gold" if len(hash_labels[audio]) != 1 else "duplicate_audio"
            continue
        seen.add(audio)
        record = refs[keys[uid]]
        validate_record(record, identity)
        if record["request_id"] != keys[uid]:
            raise ValueError("misbound reference")
        gold = labels.index(row["intent"])
        prob = record["prob"]
        rival = max((i for i in range(len(labels)) if i != gold), key=prob.__getitem__)
        gap = prob[gold] - prob[rival]
        kind = "error" if gap < 0 else "tie" if gap == 0 else "boundary" if gap < loss["margin"] else "replay"
        rows[uid] = dict(index=len(rows), gold=gold, competitor=rival, ref_gap=gap, group=row["source_group"],
                         stratum=kind, audio_sha256=audio, request_key=keys[uid])
        probs[uid] = prob
        outcomes[row["source_group"], gold].append(max(range(len(labels)), key=prob.__getitem__) == gold)
    for row in rows.values():
        group = outcomes[row["group"], row["gold"]]
        row["rerecording"] = ("single" if len(group)==1 else "all_correct" if all(group)
                              else "mixed" if any(group) else "all_wrong")
    design = config["schedule"]
    batches = shuffled_batches(rows, design)
    reward = old["bidirectional_contrast"]["reward"]
    target, schedule, counts, audit, ordering = compile_schedule(
        dict(schedule=design, reward=reward, updates=len(batches)), rows, batches, list(rows), probs)
    strata = Counter(row["stratum"] for row in rows.values())
    def summarize(fields):
        counts, groups = Counter(), defaultdict(set)
        for row in rows.values():
            key = "/".join(str(row[field]) for field in fields)
            counts[key] += 1
            groups[key].add(row["group"])
        return {k: dict(recordings=v, groups=len(groups[k])) for k,v in sorted(counts.items())}
    prior_errors = {uid for uid,row in prior.items() if row["ref_gap"] < 0}
    evidence = dict(eligible=len(eligible), strata=dict(strata), excluded=dict(Counter(excluded.values())),
                    prior_errors=len(prior_errors), prior_errors_admitted=len(prior_errors & rows.keys()),
                    prior_error_exclusions={uid:excluded[uid] for uid in sorted(prior_errors-rows.keys())},
                    class_strata=summarize(("gold","stratum")), directions=summarize(("gold","competitor")),
                    rerecordings=summarize(("rerecording",)))
    manifest = {k:old[k] for k in ("schema","source_policy","view_sha256","reference_sha256","release_sha256",
                                    "candidate_ids","calibration")}
    manifest.update(recordings=len(rows), schedule=design, updates=len(batches), batch_real=counts,
                    real_exposures=sum(counts), probability_gap=loss, strata=dict(strata),
                    population_audit=digest(evidence), intermediate_steps=[math.ceil(len(rows)/design["global_batch"])],
                    bidirectional_contrast=dict(reward=reward, batch_order_sha256=ordering, audit=audit))
    dst = Path(config["output"])
    check_paths({"gap targets":dst})
    dst.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dst.parent) as tmp:
        stage = Path(tmp)/"targets"
        stage.mkdir()
        for name,data in (("rows",rows),("batches",schedule),("calibration",calibration),("population-audit",evidence)):
            commit_json(stage/(name+".json"),data)
        commit_file(stage/"targets.pt",lambda f: torch.save(target,f))
        manifest["files"] = {p.name:file_hash(p) for p in sorted(stage.iterdir())}
        commit_json(stage/"MANIFEST.json",manifest)
        if dst.exists():
            verified(dst/"MANIFEST.json",file_hash(stage/"MANIFEST.json"))
            for name in manifest["files"]:
                pinned(dst,manifest,name)
        else:
            stage.rename(dst)
            sync_directory(dst.parent)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("local")
    args = parser.parse_args()
    config, _ = load_config(args.config,args.local)
    torch.set_num_threads(config["cpu_threads"])
    result = build(config)
    print(json.dumps({k:result[k] for k in ("recordings","updates","strata")}))

"""Compile a fixed Bernoulli view from an audited joint schedule and native cache."""
import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_reference_dataset import digest, validate_record
from multimodal_jev.context_tpo_contrast import ContrastSpec, project
from multimodal_jev.local_checkpoint import commit_file, commit_json, sync_directory
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.build_tpo_view import read, verified


def pinned(root, manifest, name):
    return verified(root / name, manifest["files"][name])


def frozen_inputs(c):
    old, ref, release = (Path(c[k]) for k in ("baseline", "reference", "release"))
    m = read(verified(old / "MANIFEST.json", c["baseline_sha256"]))
    parent = read(verified(Path(c["parent_config"]), c["parent_sha256"]))
    r = read(verified(ref / "MANIFEST.json", m["reference_sha256"]))
    clean = read(verified(release / "MANIFEST.json", m["release_sha256"]))
    identity = r["identity"]
    ids = m["candidate_ids"]
    if (m["schema"] != "jev-tpo-joint-view-v1" or m["schedule"] != c["schedule"]
            or m["updates"] != c["updates"] or m["real_exposures"] != c["recordings"] * c["schedule"]["epochs"]
            or m["source_policy"] != parent["source_policy_id"] or m["source_policy"] != identity["policy_id"]
            or m["reference_sha256"] != parent["reference_dataset"]["manifest_sha256"]
            or m["release_sha256"] != parent["release_sha256"] or m["release_sha256"] != identity["release_sha256"]
            or ids != identity["candidate_ids"] or ids != [
                x["id"] for x in read(pinned(release, clean, "tasks.json"))["intent"]["schema"]["candidates"]]):
        raise ValueError("registered schedule, source policy, candidates or reference differ")
    rows, batches, historical = (read(pinned(old, m, n)) for n in ("rows.json", "batches.json", "native-probs.json"))
    ref_pop, records = (read(pinned(ref, r, n)) for n in ("population.json", "records.json"))
    population = {}
    with pinned(release, clean, "learning.jsonl").open() as stream:
        for line in stream:
            x = json.loads(line)
            if (x["official_split"] != "train" or x["role"] != "learning" or x["known_pending_source"] is not False
                    or x["id"] in population or not x["source_group"] or x["intent"] not in ids):
                raise ValueError("expected unique admissible source TRAIN only")
            population[x["id"]] = x
    if (set(population) != set(ref_pop) or digest(ref_pop) != identity["population_sha256"]
            or len(rows) != c["recordings"] or set(rows) != set(historical) or not set(rows) <= set(population)
            or sorted(x["index"] for x in rows.values()) != list(range(len(rows)))):
        raise ValueError("reference or training population differs")
    order = sorted(rows, key=lambda uid: rows[uid]["index"])
    for uid in order:
        x, clean_row = rows[uid], population[uid]
        k = ref_pop[uid]
        r = records[k]
        validate_record(r, identity)
        if (type(x["gold"]) is not int or not 0 <= x["gold"] < len(ids)
                or k != r["request_id"] or k != x["request_key"] or r["prob"] != historical[uid]
                or ids[x["gold"]] != clean_row["intent"] or clean_row["audio_sha256"] != x["audio_sha256"]):
            raise ValueError("frozen score, input or supervision differs")
    training_groups = {population[key]["source_group"] for key in rows}
    training_hashes = {x["audio_sha256"] for x in rows.values()}
    calibration = [
        {k: x[k] for k in ("id", "source_group", "audio_sha256", "intent")}
        for key, x in sorted(population.items())
        if key not in rows and x["source_group"] not in training_groups and x["audio_sha256"] not in training_hashes
    ]
    coverage = Counter(x["intent"] for x in calibration)
    if not coverage:
        raise ValueError("empty isolated TRAIN calibration population")
    source = dict(source_policy=m["source_policy"], release_sha256=m["release_sha256"], candidate_ids=ids,
                  reference_sha256=m["reference_sha256"], view_sha256=parent["training_dataset"]["manifest_sha256"])
    return rows, batches, order, historical, calibration, coverage, source


def compile_schedule(c, rows, batches, order, probs):
    spec = ContrastSpec(**c["reward"])
    ref = torch.tensor([probs[uid] for uid in order], dtype=torch.float32)
    gold = torch.tensor([rows[uid]["gold"] for uid in order], dtype=torch.long)
    shape = (c["schedule"]["epochs"], len(rows))
    target = torch.empty((*shape, ref.shape[1]), dtype=torch.float32)
    seen = torch.zeros(shape, dtype=torch.bool)
    audit = Counter()
    expected = []
    schedule = []
    if len(batches) != c["updates"]:
        raise ValueError("incorrect schedule length")
    for i, batch in enumerate(batches):
        epoch = batch["epoch"]
        slots = batch["slots"]
        if (batch["index"] != i or not 0 <= epoch < shape[0] or batch["mode"] != "joint"
                or len(slots) != c["schedule"]["global_batch"]
                or any(type(s["padding"]) is not bool or s["weight"] != int(not s["padding"]) for s in slots)):
            raise ValueError("JEV-201 batch topology differs")
        ix = torch.tensor([rows[s["id"]]["index"] for s in slots if not s["padding"]], dtype=torch.long)
        if not len(ix) or ix.unique().numel() != len(ix) or seen[epoch, ix].any():
            raise ValueError("empty or repeated real exposure")
        q, report = project(ref[ix], gold[ix], spec)
        target[epoch, ix] = q
        seen[epoch, ix] = True
        audit.update({key: value for key, value in report.items() if key != "clipped"})
        audit.update({f"{key}_clipped": value for key, value in report["clipped"].items()})
        expected.append(len(ix))
        schedule.append(dict(index=i, epoch=epoch, slots=[
            dict(id=s["id"], padding=s["padding"], weight=s["weight"], paired=False, confusion=False)
            for s in slots]))
    if not seen.all() or not torch.isfinite(target).all():
        raise ValueError("incomplete or non-finite fixed targets")
    digest_before = digest([[b["epoch"], b["index"], [[s["id"], s["padding"]] for s in b["slots"]]] for b in batches])
    digest_after = digest([[b["epoch"], b["index"], [[s["id"], s["padding"]] for s in b["slots"]]] for b in schedule])
    if digest_before != digest_after:
        raise ValueError("batch membership or order changed")
    return target, schedule, expected, dict(audit), digest_after


def reuse_targets(c, dst):
    """Admit an independently pinned target release without projecting it again."""
    m = read(verified(dst / "MANIFEST.json", c["reuse_manifest_sha256"]))
    contrast = m["bidirectional_contrast"]
    baseline = read(verified(Path(c["baseline"]) / "MANIFEST.json", c["baseline_sha256"]))
    parent = read(verified(Path(c["parent_config"]), c["parent_sha256"]))
    verified(Path(c["reference"]) / "MANIFEST.json", m["reference_sha256"])
    verified(Path(c["release"]) / "MANIFEST.json", m["release_sha256"])
    if (m["schema"] != "jev-tpo-reward-view-v1"
            or m["source_policy"] != baseline["source_policy"]
            or m["view_sha256"] != parent["training_dataset"]["manifest_sha256"]
            or m["reference_sha256"] != baseline["reference_sha256"]
            or m["release_sha256"] != baseline["release_sha256"]
            or m["candidate_ids"] != baseline["candidate_ids"]
            or any(m[name] != c[name] for name in ("schedule", "recordings", "updates"))
            or any(contrast[name] != c[name] for name in ("baseline_sha256", "parent_sha256", "reward"))
            or set(m["files"]) != {"rows.json", "batches.json", "calibration.json", "targets.pt"}):
        raise ValueError("audited training targets do not match the requested inputs")
    for name in m["files"]:
        pinned(dst, m, name)
    return m


def build(c):
    dst = Path(c["output"])
    if "reuse_manifest_sha256" in c:
        return reuse_targets(c, dst)
    check_paths({"bidirectional_targets": dst})
    torch.set_num_threads(c["cpu_threads"])
    rows, batches, order, probs, calibration, coverage, source = frozen_inputs(c)
    target, schedule, real, audit, ordering = compile_schedule(c, rows, batches, order, probs)
    base = dict(**source, schema="jev-tpo-reward-view-v1", schedule=c["schedule"],
                recordings=len(rows), updates=len(schedule), real_exposures=sum(real), batch_real=real,
                bidirectional_contrast=dict(baseline_sha256=c["baseline_sha256"], parent_sha256=c["parent_sha256"],
                                            reward=c["reward"], batch_order_sha256=ordering, audit=audit),
                calibration=dict(recordings=len(calibration), by_class={key: coverage[key] for key in source["candidate_ids"]},
                                 absent_classes=[key for key in source["candidate_ids"] if not coverage[key]],
                                 split="source_clean_train", released_before_endpoints=True))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".bidirectional-", dir=dst.parent) as tmp:
        stage = Path(tmp) / "targets"
        stage.mkdir()
        for name, content in (("rows.json", rows), ("batches.json", schedule), ("calibration.json", calibration)):
            commit_json(stage / name, content)
        commit_file(stage / "targets.pt", lambda f: torch.save(target, f))
        base["files"] = {p.name: file_hash(p) for p in sorted(stage.iterdir())}
        commit_json(stage / "MANIFEST.json", base)
        if dst.exists():
            verified(dst / "MANIFEST.json", file_hash(stage / "MANIFEST.json"))
            for name in base["files"]:
                pinned(dst, base, name)
        else:
            stage.rename(dst)
            sync_directory(dst.parent)
    return base


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("local")
    args = parser.parse_args()
    result = build(load_config(args.config, args.local)[0])
    print({k: result[k] for k in ("recordings", "updates", "real_exposures", "bidirectional_contrast")})

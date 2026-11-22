"""Build a private per-intent support bank from pinned official TRAIN text annotations."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.intent_card_template import compile_intent_card_request
from multimodal_jev.context_decision_contract import render_prefix_context_request


def read(path, digest):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"input changed: {path.name}")
    return data


def build(config, clean, annotations, roster_path, truth_path):
    pins = config["pins"]
    manifest = json.loads(read(clean / "MANIFEST.json", pins["clean_manifest"]))
    task = json.loads(read(clean / "tasks.json", manifest["files"]["tasks.json"]))["intent"]
    labels = [r["id"] for r in task["schema"]["candidates"]]
    if set(labels) != set(config["activation"]):
        raise ValueError("card definitions differ from the candidate inventory")
    admitted = {r["id"]: r for line in read(clean / "learning.jsonl", manifest["files"]["learning.jsonl"]).splitlines()
                for r in [json.loads(line)]}
    examples = defaultdict(dict)
    source = {}
    for line in read(annotations, pins["annotations"]).splitlines():
        row = json.loads(line)
        for recording in row["recordings"]:
            source["slurp:" + recording["file"]] = row
    roster = json.loads(read(roster_path, pins["roster"]))
    truth = json.loads(read(truth_path, pins["truth"]))
    if set(truth) != {r["id"] for r in roster}:
        raise ValueError("reference and roster differ")
    rivals = defaultdict(Counter)
    text_labels = defaultdict(set)
    for asset in sorted(roster, key=lambda r: r["id"]):
        uid = asset["id"]
        row, ann = admitted[uid], source[uid]
        gold = row["intent"]
        if (gold != asset["intent"] or (row["scenario"],row["action"]) != (ann["scenario"],ann["action"]) or row["role"] != "learning"
                or row["official_split"] != "train" or row["known_pending_source"] is not False
                or any(row[k] != asset[k] for k in ("source_group", "audio_sha256"))):
            raise ValueError("TRAIN admission or native label mismatch")
        ref = truth[uid]
        if labels[ref["gold"]] != gold or ref["audio_sha256"] != row["audio_sha256"]:
            raise ValueError("reference row mismatch")
        if ref["stratum"] == "error":
            rivals[gold][labels[ref["competitor"]]] += 1
        key = " ".join(ann["sentence"].casefold().split())
        text_labels[key].add(gold)
        examples[gold].setdefault(key, dict(text=ann["sentence"], id=uid, group=row["source_group"],
                                            audio_sha256=row["audio_sha256"]))
    def grams(text):
        words = re.findall(r"[a-z]+", text)
        return {a + " " + b for a, b in zip(words, words[1:])}
    counts = {k: Counter(g for t in examples[k] for g in grams(t)) for k in labels}
    total = sum(counts.values(), Counter())
    cards = []
    for label in labels:
        texts = examples[label]
        cues = [g for g, n in counts[label].items() if n >= config["min_cue_texts"]]
        cues.sort(key=lambda g: (-counts[label][g] / total[g], -counts[label][g], g))
        cues = cues[:config["cue_limit"]]
        eligible = {k:v for k,v in texts.items() if len(text_labels[k]) == 1}
        def typical(k):
            return (-sum(g in grams(k) for g in cues), hashlib.sha256(f"{config['seed']}|{k}".encode()).digest())
        chosen = []
        for key in sorted(eligible, key=typical):
            if all(eligible[key]["group"] != row["group"] for row in chosen):
                chosen.append(eligible[key])
            if len(chosen) == config["examples_per_class"]:
                break
        if len(chosen) != config["examples_per_class"]:
            raise ValueError(f"insufficient distinct admitted supports for {label}")
        rival = min(rivals[label], key=lambda k: (-rivals[label][k], k)) if rivals[label] else None
        cards.append(dict(contrast_with=rival, contrast_support=rivals[label][rival],id=label, activation=config["activation"][label], distinct_train_texts=len(texts),
                          cues=[dict(text=g, within=counts[label][g], total=total[g]) for g in cues], examples=chosen))
    active = set()
    for rule in config["semantic_rules"]:
        members = rule["labels"]
        if len(set(members)) != len(members) or len(members) < 2 or set(members) - set(labels) or not rule["rule"].strip():
            raise ValueError("invalid semantic contrast")
        active.update(members)
    if set(config["unresolved_rules"]) != set(labels) - active:
        raise ValueError("unaccounted semantic rule coverage")
    return dict(schema=config["schema"], pins=pins, cards=cards,
                semantic_rules=config["semantic_rules"], unresolved_rules=config["unresolved_rules"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "clean", "annotations", "roster", "truth", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--template", type=Path, help="also render the configured fixed prefixes beside the bank")
    args = parser.parse_args()
    bank = build(json.loads(args.config.read_bytes()), args.clean, args.annotations, args.roster, args.truth)
    if args.output.exists() and bank != json.loads(args.output.read_bytes()):
        raise ValueError("publish to a new snapshot")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        commit_json(args.output, bank)
    if args.template:
        manifest = json.loads(read(args.clean / "MANIFEST.json", bank["pins"]["clean_manifest"]))
        task = json.loads(read(args.clean / "tasks.json", manifest["files"]["tasks.json"]))["intent"]
        template = json.loads(args.template.read_bytes())
        panels = {}
        for arm in template["arms"]:
            request = compile_intent_card_request("audio", task, bank, template, arm)
            prefix, _, readouts = render_prefix_context_request(request)
            panels[arm] = dict(prefix=prefix, readouts=readouts, candidate_ids=[q.id for q in request.questions])
        target = args.output.with_name(args.output.stem + "-prefixes.json")
        if target.exists() and panels != json.loads(target.read_bytes()):
            raise ValueError("publish changed prefixes to a new snapshot")
        if not target.exists():
            commit_json(target, panels)

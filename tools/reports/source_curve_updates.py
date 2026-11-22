"""Measure checkpoint-to-checkpoint parameter displacement on CPU."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def read_policy(path, policy_id):
    manifest = json.loads((path / "MANIFEST.json").read_bytes())
    weights = path / "execution.pt"
    with weights.open("rb") as src:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            checksum.update(chunk)
        digest = checksum.hexdigest()
    if manifest["policy_id"] != policy_id or digest != manifest["files"][weights.name]:
        raise ValueError(f"checkpoint identity mismatch at {path}")
    return torch.load(weights, weights_only=True, map_location="cpu")


def displacement(left, right):
    if left.keys() != right.keys():
        raise ValueError("parameter names differ across checkpoints")
    totals = {}
    for name, tensor in left.items():
        if tensor.shape != right[name].shape:
            raise ValueError(f"parameter shape changed: {name}")
        if ".lora_" in name:
            group = "lora"
        elif name.startswith("decision_head."):
            group = "decision_head"
        elif name == "decision_token_row":
            group = "marker"
        else:
            raise ValueError(f"unknown parameter: {name}")
        initial = tensor.double()
        delta = right[name].double() - initial
        part = totals.setdefault(group, [0.0, 0.0])
        part[0] += delta.square().sum().item()
        part[1] += initial.square().sum().item()
    return {group: (change / initial) ** 0.5 for group, (change, initial) in totals.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs=2, action="append", required=True, metavar=("PATH", "POLICY_ID"))
    parser.add_argument("--cpu-threads", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    if len(args.checkpoint) < 2:
        parser.error("provide at least two checkpoints in chronological order")
    states = [read_policy(Path(path), policy_id) for path, policy_id in args.checkpoint]
    print(json.dumps([displacement(a, b) for a, b in zip(states, states[1:])]))

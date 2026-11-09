#!/usr/bin/env python3
"""Read explicit ordered rollout/actor/export paths and write a separate audit."""
import argparse
import json
from pathlib import Path
from multimodal_jev.trajectory_budget import audit_trajectory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', type=Path, required=True,
                        help='JSON list of objects with rollout, actor and export paths')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit_trajectory(json.loads(args.stages.read_text()))
    # Never overwrite historical evidence or a concurrently written audit.
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')

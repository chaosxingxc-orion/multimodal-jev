#!/usr/bin/env python3
"""Verify completed actor artifacts without loading the base model or updating it."""
import argparse
import json
from pathlib import Path
from multimodal_jev.actor_audit import audit_actor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit_actor(args.run)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')

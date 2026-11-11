#!/usr/bin/env python3
"""Prepare clean TRAIN source bundles without task projection or optimization."""
import argparse
import json
from pathlib import Path

from multimodal_jev.slurp_clean_sources import publish

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('clean', 'native', 'split-root', 'evidence', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.clean, args.native, args.split_root, args.evidence, args.output), indent=2))

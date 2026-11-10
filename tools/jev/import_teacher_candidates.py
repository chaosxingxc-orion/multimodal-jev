#!/usr/bin/env python3
"""Stage complete teacher proposals without executing or promoting them."""
import argparse
import json
from pathlib import Path
from multimodal_jev.teacher_candidates import stage_candidates

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(stage_candidates(args.request, args.output), indent=2))

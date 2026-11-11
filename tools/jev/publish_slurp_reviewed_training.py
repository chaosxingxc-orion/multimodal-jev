#!/usr/bin/env python3
"""Publish reviewed decisions; never creates or approves review labels."""
import argparse
import json
from pathlib import Path

from multimodal_jev.slurp_training_release import publish

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('source','source-receipt','tasks','ledger','destination'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    path, report = publish(args.source,args.source_receipt,args.tasks,args.ledger,args.destination)
    print(json.dumps(dict(path=str(path),report=report),indent=2))

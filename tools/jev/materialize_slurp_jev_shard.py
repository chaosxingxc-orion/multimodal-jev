#!/usr/bin/env python3
"""Expand exactly one full-task shard from a pinned whole-split plan."""
import argparse
import json
from pathlib import Path
from multimodal_jev.slurp_jev_shards import materialize_shard

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--shard',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(materialize_shard(a.plan,a.shard,a.output),indent=2))

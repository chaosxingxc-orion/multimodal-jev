#!/usr/bin/env python3
"""Materialize one hash-bound SFC-Jev input shard without reading targets."""
import argparse
import json
from pathlib import Path
from multimodal_jev.sfc_jev_shards import materialize

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('plan','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('plan-hash','shard-id'):p.add_argument('--'+k,required=True)
    a=p.parse_args();print(json.dumps(materialize(a.plan,a.plan_hash,a.shard_id,a.output),indent=2))

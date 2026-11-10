#!/usr/bin/env python3
"""Create a full-split label-free execution plan; no model dispatch."""
import argparse
import json
from pathlib import Path

from multimodal_jev.consumption import canonical
from multimodal_jev.slurp_jev_shards import plan_shards


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True)
    p.add_argument('--split',choices=['train','valid','test'],required=True)
    p.add_argument('--recordings-per-shard',type=int,default=100)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=plan_shards(a.release,a.split,a.recordings_per_shard)
    with a.output.open('xb') as handle:handle.write(canonical(result))
    print(json.dumps({k:v for k,v in result.items() if k!='shards'},indent=2))

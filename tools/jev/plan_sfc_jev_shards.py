#!/usr/bin/env python3
"""Plan every SFC-Jev decision with indivisible conversations; no inference."""
import argparse
import json
from pathlib import Path
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_jev_shards import make_plan

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--split',choices=('train_id','test_id','test_ood'),required=True)
    p.add_argument('--decisions-per-shard',type=int,default=1024)
    p.add_argument('--partition',type=Path);p.add_argument('--partition-hash')
    p.add_argument('--part',choices=('learning','selection'));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=make_plan(a.release,a.split,a.decisions_per_shard,partition_path=a.partition,partition_hash=a.partition_hash,part=a.part)
    with a.output.open('xb') as handle:handle.write(canonical(result))
    print(json.dumps({k:v for k,v in result.items() if k!='shards'},indent=2))

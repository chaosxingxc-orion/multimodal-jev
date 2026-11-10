#!/usr/bin/env python3
"""Register one SFC teacher call after accepted learning feedback, without dispatch."""
import argparse
import json
from pathlib import Path
from multimodal_jev.sfc_teacher_wire import configure

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for k in ('root','reference','executor'):
        p.add_argument('--'+k,type=Path,required=True)
    for k in ('prepared-hash','partition-hash'):
        p.add_argument('--'+k,required=True)
    a=p.parse_args()
    print(json.dumps(configure(a.root,a.reference,a.executor,a.prepared_hash,a.partition_hash),indent=2))

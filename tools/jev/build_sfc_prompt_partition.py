#!/usr/bin/env python3
"""Register immutable whole-TRAIN prompt-learning/selection boundaries."""
import argparse
import json
from pathlib import Path
from multimodal_jev.sfc_partition import build
from multimodal_jev.consumption import canonical

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('release','preflight','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--exposure-report',type=Path,action='append',default=[])
    a=p.parse_args();report=build(a.release,a.preflight,a.exposure_report)
    with a.output.open('xb') as f:f.write(canonical(report))
    print(json.dumps(report['summary'],indent=2))

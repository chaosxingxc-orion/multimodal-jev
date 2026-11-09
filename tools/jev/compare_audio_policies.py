#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from multimodal_jev.compare_policies import compare

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('manifest','base','candidate','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--draws',type=int,default=2000);p.add_argument('--seed',type=int,default=17)
    a=p.parse_args()
    result=compare(a.manifest,a.base,a.candidate,a.draws,a.seed)
    with a.output.open('x') as handle:json.dump(result,handle,indent=2)

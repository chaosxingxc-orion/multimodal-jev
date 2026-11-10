#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from multimodal_jev.consumption import export_release

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--native-only',action='store_true')
    args=parser.parse_args()
    output,receipt=export_release({s:args.source/(s+'.jsonl') for s in ('train','valid','test')},
        args.output,args.dataset,args.native_only)
    print(json.dumps({'output':str(output),**receipt},indent=2))

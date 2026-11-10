#!/usr/bin/env python3
"""Stage local SFC teacher candidate data without execution or promotion."""
import argparse
import json
from pathlib import Path
from multimodal_jev.sfc_prompt_candidates import stage

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('proposal','feedback','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('proposal-hash','feedback-hash'):p.add_argument('--'+k,required=True)
    a=p.parse_args()
    print(json.dumps(stage(a.proposal,a.feedback,a.output,a.proposal_hash,a.feedback_hash),indent=2))

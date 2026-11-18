"""Retired development dispatcher: classified startup gates must precede release.

The earlier draft is preserved in immutable diagnostic snapshots. It has no
training authorization: use startup_category_selfcheck to produce CPU evidence.
"""
import argparse
from multimodal_jev.local_config import load_config


def run(config,provenance):
    raise RuntimeError('startup categories C5/C6/C7 are not admitted; all-stage composed lifecycle and fixed-image/NPU evidence required')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    args=parser.parse_args();config,provenance=load_config(args.config,args.local)
    run(config,provenance)

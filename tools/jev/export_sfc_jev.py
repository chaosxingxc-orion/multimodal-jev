#!/usr/bin/env python3
"""Export full SFC Jev bundles, preserving all native splits and call arguments."""
import argparse
import json
from pathlib import Path
from multimodal_jev.sfc_jev import export

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args()
    path, report = export(a.source, a.output_root)
    print(json.dumps(dict(path=str(path), **report), indent=2))

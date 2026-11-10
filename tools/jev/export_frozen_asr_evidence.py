#!/usr/bin/env python3
"""Lock existing ASR outputs before evaluating replacement prompts."""
import argparse
from pathlib import Path
from multimodal_jev.baseline import file_hash
from multimodal_jev.frozen_evidence import export_evidence

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = export_evidence(args.run, args.output)
    print({'path': str(args.output), 'sha256': file_hash(args.output),
           'requests': len(result['requests']), 'recordings': len(result['records'])})

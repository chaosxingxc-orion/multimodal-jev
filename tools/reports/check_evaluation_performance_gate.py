"""Check pinned performance evidence; no model calls or remote dispatch."""
import argparse
import json

from multimodal_jev.local_config import load_config
from multimodal_jev.evaluation_performance_gate import require_performance_admission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    args = parser.parse_args()
    config, provenance = load_config(args.config, args.local)
    result = require_performance_admission(config)
    print(json.dumps(dict(result, provenance=provenance), indent=2))


if __name__ == '__main__':
    main()

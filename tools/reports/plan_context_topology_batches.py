"""Resolve portable topology settings and write a derived, non-dispatch plan."""
import argparse
import json

from multimodal_jev.context_topology_batch_plan import derive
from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    args = parser.parse_args()
    c, provenance = load_config(args.config, args.local)
    plans = {name: derive(arm) for name, arm in c['arms'].items()}
    result = dict(schema='jev-topology-batch-study-plan-v1', arms=plans,
                  provenance=provenance, training_dispatch_allowed=False)
    commit_json(c['output'], result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

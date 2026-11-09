#!/usr/bin/env python3
"""Check real audio RL assets without importing torch or launching an engine."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.rl_preflight import validate_schedule


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('preflight receipt already exists')
    started=time.monotonic()
    source=Path(__file__).parents[2]
    config=json.loads(args.config.read_text())
    receipt={'config_path':str(args.config.resolve()),'config_sha256':file_hash(args.config),
             'config':config,'checked_at_utc':datetime.now(timezone.utc).isoformat(),
             'source_sha256':{name:file_hash(source/name) for name in (
                 'tools/jev/preflight_audio_rl.py','src/multimodal_jev/rl_preflight.py',
                 'src/multimodal_jev/policy_identity.py','src/multimodal_jev/pair_reward.py',
                 'src/multimodal_jev/rollout.py','src/multimodal_jev/task_schema.py')},
             'scope':'artifact and TRAIN selection checks only; no engine, optimizer update or efficacy evaluation'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        receipt.update(validate_schedule(config),status='passed')
    except Exception as exc:
        receipt.update(status='failed',error=repr(exc),elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output,receipt)
        raise
    receipt['elapsed_seconds']=time.monotonic()-started
    atomic_json(args.output,receipt)
    print(json.dumps({key:receipt[key] for key in (
        'status','iterations','unique_examples','unique_audio','scheduled_responses',
        'initial_policy_id','actor_base_policy_id','elapsed_seconds','scope')},indent=2))


if __name__=='__main__':main()

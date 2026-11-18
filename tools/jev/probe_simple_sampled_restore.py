#!/usr/bin/env python3
"""Fresh-process DP4 restore of independently accepted sampled RL state."""
import argparse,json,os,time
from multimodal_jev.baseline import atomic_json
from multimodal_jev.local_runtime_storage import require_training_storage
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();config=json.loads(a.config.read_text());rank=int(os.environ['RANK'])
    if (int(os.environ['WORLD_SIZE'])!=4 or int(os.environ['LOCAL_RANK'])!=rank
            or config['sampled_probe'].get('fixture_initialization')!='restore_updated_rl_only'
            or config['sampled_probe']['algorithm'] not in ('grpo_v1','rloo_v1')):
        raise ValueError('explicit updated policy recovery and fixed DP4 required')
    require_training_storage(a.output)
    out=a.output/f'rank-{rank}';out.mkdir(parents=True,exist_ok=False)
    from probe_simple_sampled_npu import restore_accepted_fixture
    try:
        restore_accepted_fixture(config,a.output,rank,restore_only=True)
        atomic_json(out/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(out/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise

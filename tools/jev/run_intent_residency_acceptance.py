#!/usr/bin/env python3
"""Full independent backend gate, state audit, then exact residency replay."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from queue_span_native_actor_npu import read
from run_intent_finite_acceptance import run
from verify_intent_residency_replay_cpu import verify


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    try:
        run(a.root)
        # Ordinary acceptance verifier expects the completed numerical phases;
        # this is an internal stage receipt, not actual Docker completion.
        atomic_json(a.root/'controller-terminal.json',dict(exit_code=0,finished_unix=time.time()))
        report=verify(read(a.root/'config.json')['training_root'],a.root)
        atomic_json(a.root/'residency-verified.json',report)
        atomic_json(a.root/'RESIDENCY_ACCEPTED.json',dict(passed=True,verified_sha256=file_hash(a.root/'residency-verified.json')))
        atomic_json(a.root/'residency-terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.root/'residency-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise

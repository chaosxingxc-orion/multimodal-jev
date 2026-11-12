#!/usr/bin/env python3
"""Publish all registered D/Q/P learning effects after exact P12 acceptance."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from queue_span_native_actor_npu import read
from queue_span_dp4_resume import check_pins
from run_intent_lr_control import await_qp
from compare_intent_matched_pilots import run as compare


def run(config,out):
    check_pins(config['pins']);path=Path(config['comparison_config'])
    if config['pins'].get(str(path))!=file_hash(path):raise ValueError('fixed complete comparison required')
    comparison=read(path);check_pins(comparison['pins'])
    if (comparison['image']!=config['image'] or set(comparison['arms'])!={'D','Q','P'}
            or any(len(v)!=12 for v in comparison['arms'].values())
            or comparison['arms']['P'][-1]['suite_root']!=config['predecessor_suite']):
        raise ValueError('all registered twelve-step arms required')
    await_qp(config,out)
    result=compare(comparison)
    if (result.get('complete') is not True or result['role']!='learning'
            or result['updates']!=12 or result['recordings']!=89
            or [p['update'] for p in result['curve']]!=list(range(13))):
        raise ValueError('complete fixed learning curves required')
    atomic_json(out/'comparison.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output);atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time(),scope='Complete learning effects, not independent efficacy.'))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise

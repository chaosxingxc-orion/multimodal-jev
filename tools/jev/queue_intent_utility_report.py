#!/usr/bin/env python3
"""Read-only complete objective report after the exact registered controller."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read
from wait_intent_priority_tail import wait_tail
from report_intent_utility_curves import run as report


def run(config,out):
    check_pins(config['pins'])
    path=Path(config['protocol']);predecessor=Path(config['priority_predecessor'])
    if (config['priority_kind']!='utility_curves12'
            or config['pins'].get(str(path))!=file_hash(path)
            or config['protocol_sha256']!=file_hash(path)
            or config['pins'].get(str(predecessor/'config.json'))!=file_hash(predecessor/'config.json')):
        raise ValueError('pinned exact full utility predecessor and protocol required')
    previous=read(predecessor/'config.json')
    if (Path(previous['plan_root'])/'protocol.json'!=path or previous['image']!=config['image']):
        raise ValueError('predecessor targets another registered study')
    wait_tail(config,out)  # Exact process while waiting; actual36 pairs/replay when done.
    result=report(path,config['protocol_sha256'])
    expected={a['name'] for a in read(path)['arms']}
    if (result.get('complete') is not True or result['role']!='learning'
            or result['updates']!=12 or result['protocol_sha256']!=config['protocol_sha256']
            or set(result['curves'])!=expected
            or any([p['update'] for p in points]!=list(range(1,13)) for points in result['curves'].values())):
        raise ValueError('complete registered objective report required')
    atomic_json(out/'report.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output)
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time(),
            scope='Full objective diagnostics only; not held-out efficacy.'))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise

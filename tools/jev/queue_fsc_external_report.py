#!/usr/bin/env python3
"""Read-only report after the original complete FSC matrix; poll every600s."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read
from compare_fsc_external_reference import run as compare


def run(config,out):
    root=Path(config['predecessor']);check_pins(config['pins']);receipt=read(root/'launch.json')
    for name in ('config.json','launch.json'):
        if config['pins'].get(str(root/name))!=file_hash(root/name):raise ValueError('pinned original controller required')
    while True:
        check_pins(config['pins']);terminal=root/'run/terminal.json'
        if terminal.exists():
            if read(terminal)['exit_code']!=0:raise ValueError('original matrix failed')
            prior=read(root/'config.json');protocol=Path(prior['matrix_root'])/'protocol.json'
            pins=read(root/'run/matrix-pins.json');digest=pins.get(str(protocol))
            if not digest or file_hash(protocol)!=digest:raise ValueError('original dispatched matrix protocol required')
            saved=read(root/'run/comparison.json')
            expected={'step0'}|{f'{a}-step{s}' for a in ('D','Q','P') for s in (6,12)}
            if saved['protocol_sha256']!=digest or set(saved['policies'])!=expected:raise ValueError('complete scored predecessor required')
            check_pins(saved['files'])
            # Full actual container/gate checks repeated by compare before gold.
            atomic_json(out/'comparison.json',compare(protocol,digest,config['reference_config']))
            return
        try:actual=(Path('/proc')/str(receipt['pid'])/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except FileNotFoundError:
            if terminal.exists():continue
            raise ValueError('original controller missing without terminal')
        if actual!=receipt['argv']:raise ValueError('predecessor PID reused')
        atomic_json(out/'progress.json',dict(stage='waiting_complete_fsc_matrix',pid=receipt['pid'],updated_unix=time.time()))
        time.sleep(600)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output);atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise

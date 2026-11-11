#!/usr/bin/env python3
"""Bounded learning-only SFT with explicit physical-device admission."""
import argparse
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        import torch
        import torch_npu
        from multimodal_jev.sft import train
        config=json.loads(args.config.read_text())
        nodes=sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))
        if nodes!=['davinci4'] or torch.npu.device_count()!=1:
            raise ValueError('pilot requires isolated physical NPU 4 only')
        torch.npu.set_device(0)
        torch.npu.manual_seed_all(config['seed'])
        if file_hash(config['train_manifest'])!=config['expected_manifest_sha256']:
            raise ValueError('pilot training manifest changed')
        atomic_json(args.output/'admission.json',dict(device_nodes=nodes,
            logical_device='npu:0',config_sha256=file_hash(args.config),observed_unix=time.time()))
        train(config,args.output/'sft')
        atomic_json(args.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()))
        raise


if __name__=='__main__':main()

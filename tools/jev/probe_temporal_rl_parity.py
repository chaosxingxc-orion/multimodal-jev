#!/usr/bin/env python3
"""Run sampler then three independent actor audits, without parameter updates."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from multimodal_jev.baseline import atomic_json, file_hash


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    config=json.loads(args.config.read_text())
    try:
        nodes=sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))
        if nodes!=['davinci4']:raise ValueError('isolated physical NPU 4 required')
        for path,digest in config['pins'].items():
            if file_hash(path)!=digest:raise ValueError('changed input: '+path)
        here=Path(__file__).parent
        commands=[('sampling',[sys.executable,'-B',str(here/'sample_audio_policy.py'),
            '--config',config['sampling_config'],'--output',str(args.output/'sampling')])]
        for mode in ('merged_full','merged_cached','adapter_full'):
            argv=[sys.executable,'-B',str(here/'audit_rollout_logprobs.py'),
                '--rollout',str(args.output/'sampling'),'--output',str(args.output/mode),
                '--device','npu:0','--dtype','bfloat16']
            if mode=='merged_cached':argv+=['--cached-decode']
            if mode=='adapter_full':argv+=['--base',config['base'],'--adapter',config['adapter']]
            commands.append((mode,argv))
        outcomes={}
        for name,argv in commands:
            atomic_json(args.output/'progress.json',dict(phase=name,heartbeat_unix=time.time()))
            atomic_json(args.output/(name+'-command.json'),dict(argv=argv))
            with (args.output/(name+'.log')).open('w') as log:
                process=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=480)
            outcomes[name]=dict(exit_code=process.returncode)
            result=args.output/name/'result.json'
            if result.exists():outcomes[name]['result']=json.loads(result.read_text())
            if name=='sampling' and process.returncode:
                raise RuntimeError('sampler failed; preserve logs and do not audit incomplete records')
            # A parity failure is a measured result; complete other diagnostic arms.
        atomic_json(args.output/'result.json',dict(status='diagnostic_complete',outcomes=outcomes,
            threshold=0.02,optimizer_updates=0,
            scope='Six-prompt SLURP probability diagnosis. No batched/distributed actor admission or task improvement claim.'))
        atomic_json(args.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise


if __name__=='__main__':main()

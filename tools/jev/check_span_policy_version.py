#!/usr/bin/env python3
"""Sequential HF reload then vLLM parity for one immutable policy version.

Subprocess boundaries release accelerator memory between backends. The root
is exclusive and READY is written only after all requested inputs pass.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from multimodal_jev.baseline import atomic_json, file_hash


def run(config, output):
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:
            raise ValueError('pinned policy input changed: '+path)
    for phase,script in (('hf','build_span_hf_reference.py'),('vllm','check_span_runtime_suite.py')):
        phase_config={**config,'pins':dict(config['pins'])}
        if phase=='vllm':
            reference=output/'hf/result.json'
            if json.loads((output/'hf/terminal.json').read_text())['exit_code']!=0:
                raise ValueError('independent HF reference failed')
            phase_config['hf_results']=str(reference)
            phase_config['pins'][str(reference)]=file_hash(reference)
        path=output/f'{phase}-config.json'
        atomic_json(path,phase_config)
        subprocess.run([sys.executable,str(Path(__file__).with_name(script)),
                        '--config',str(path),'--output',str(output/phase)],check=True)
    result=json.loads((output/'vllm/result.json').read_text())
    if json.loads((output/'vllm/terminal.json').read_text())['exit_code']!=0 or not result['passed']:
        raise ValueError('policy version parity failed')
    atomic_json(output/'READY.json',dict(status='passed',base_id=result['base_id'],adapter_id=result['adapter_id'],
        recordings=len(result['records']),runtime_profile=result['runtime_profile'],
        result_sha256=file_hash(output/'vllm/result.json'),
        scope='Independent HF reload/vLLM parity for this immutable version; no optimization or generalization claim'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:
        run(json.loads(args.config.read_text()),args.output)
        atomic_json(args.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise

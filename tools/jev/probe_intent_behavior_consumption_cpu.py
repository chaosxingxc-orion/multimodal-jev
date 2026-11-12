#!/usr/bin/env python3
"""Re-encode real audio and consume same-policy behavior through actor prepare."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from queue_span_native_actor_npu import read
from queue_span_dp4_resume import check_pins


def run(config,out):
    import torch
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.intent_actor_inputs import prepare
    from multimodal_jev.intent_behavior_replay import compare_batches
    from multimodal_jev.intent_resume import authenticate
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU input consumer only')
    check_pins(config['pins'])
    for key in ('reference','candidate'):
        if config['pins'].get(config[key])!=file_hash(config[key]):raise ValueError('pinned input configs required')
    old,new=[read(config[k]) for k in ('reference','candidate')]
    allowed={'pins','resident_behavior','acceptance_root'}
    if {k:v for k,v in old.items() if k not in allowed}!={k:v for k,v in new.items() if k not in allowed}:
        raise ValueError('only same-policy behavior binding may change')
    if config['mode']=='resident':
        if 'resident_behavior' not in new or 'resident_behavior' in old:raise ValueError('explicit new resident behavior required')
    elif config['mode']=='historical_reference':
        if config['reference']!=config['candidate'] or 'resident_behavior' in new:raise ValueError('explicit historical control required')
    else:raise ValueError('explicit probe mode required')
    for cfg in (old,new):
        check_pins(cfg['pins'])
        if cfg['step']>0:authenticate(cfg)
    processor=Qwen3ASRProcessor.from_pretrained(old['model_path'],local_files_only=True)
    policy=checkpoint_identity(old['model_path'])[0]+':'+checkpoint_identity(old['initial_adapter'])[0]
    results=[]
    # Twelve original schedule positions cover all 89 sources (96 uses), with
    # unchanged labels/weights. This is re-encoding, not twelve training updates.
    for step in range(12):
        left,selection=prepare(old,processor,policy,step)
        right,other=prepare(new,processor,policy,step)
        results.append(dict(schedule_step=step,**compare_batches(left,right,selection,other)))
        atomic_json(out/'progress.json',dict(completed_batches=step+1,updated_unix=time.time()))
    unique={uid for row in results for uid in row['request_ids']}
    if len(unique)!=89:raise ValueError('complete89-source consumer population required')
    return dict(passed=True,mode=config['mode'],policy_id=policy,recordings=len(unique),source_uses=sum(r['requests'] for r in results),
        batches=results,scope='Actual audio/processor and actor prepare consumption. No training, model inference or efficacy proof.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'result.json',run(read(a.config),a.output));atomic_json(a.output/'terminal.json',dict(exit_code=0))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise

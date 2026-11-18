#!/usr/bin/env python3
"""Bounded same-container NPU context-sharing probe, without model weights.

One service process sees one runtime device. Four actor processes see the four
mounted devices. All physical IDs are read from CANN through the fixed vLLM
platform; environment indices alone are never treated as physical evidence.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(out,role,rank):
    import torch
    import torch_npu
    from vllm.platforms import current_platform
    torch.set_num_threads(1)
    logical=0 if role=='service' else rank
    expected=7 if role=='service' else 4+rank
    torch.npu.set_device(logical)
    count=torch.npu.device_count()
    physical=current_platform.visible_device_id_to_physical_device_id(logical)
    if count!=(1 if role=='service' else 4) or physical!=expected:
        raise ValueError(f'actual runtime mapping differs: count={count}, physical={physical}')
    x=torch.ones((32,32),device=torch.device('npu',logical),dtype=torch.bfloat16)
    y=x@x;torch.npu.synchronize()
    if not torch.equal(y.cpu(),torch.full((32,32),32.,dtype=torch.bfloat16)):
        raise ValueError('device compute differs')
    report=dict(role=role,rank=rank,logical=logical,physical=physical,device_count=count,
                visible=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),pid=os.getpid(),compute_exact=True)
    name='service-ready.json' if role=='service' else f'actor-{rank}.json'
    (out/name).write_text(json.dumps(report,indent=2)+'\n')
    # Keep both allocation and context alive until all actor processes finish.
    if role=='service':
        deadline=time.monotonic()+180
        while not (out/'release-service').exists():
            if time.monotonic()>deadline:raise TimeoutError('bounded probe service deadline')
            time.sleep(.2)
        if not torch.equal((x@x).cpu(),y.cpu()):raise ValueError('shared context changed result')
    torch.npu.synchronize()


def run(out,service_visible):
    from multimodal_jev.local_runtime_storage import require_training_storage
    require_training_storage(out)
    if sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))!=['davinci4','davinci5','davinci6','davinci7']:
        raise ValueError('exact physical4-7 container required')
    out.mkdir(exist_ok=False)
    children=[];streams=[];terminal=dict(exit_code=1)
    def launch(role,rank,visible):
        env=dict(os.environ)
        env.pop('ASCEND_RT_VISIBLE_DEVICES',None)
        if visible is not None:env['ASCEND_RT_VISIBLE_DEVICES']=visible
        stream=(out/f'{role}-{rank}.log').open('w');streams.append(stream)
        p=subprocess.Popen([sys.executable,__file__,'--output',str(out),'--role',role,'--rank',str(rank)],
                           env=env,stdout=stream,stderr=subprocess.STDOUT)
        children.append(p);return p
    try:
        service=launch('service',0,service_visible)
        deadline=time.monotonic()+120
        while not (out/'service-ready.json').exists():
            if service.poll() is not None:raise RuntimeError('probe service initialization failed')
            if time.monotonic()>deadline:raise TimeoutError('probe service readiness deadline')
            time.sleep(.2)
        actors=[launch('actor',rank,None) for rank in range(4)]
        deadline=time.monotonic()+120
        for actor in actors:
            if actor.wait(timeout=max(.1,deadline-time.monotonic()))!=0:
                raise RuntimeError('probe actor initialization/compute failed')
        if service.poll() is not None:raise RuntimeError('service did not remain live during actor execution')
        (out/'release-service').touch()
        if service.wait(timeout=30)!=0:raise RuntimeError('service shutdown compute failed')
        report=dict(service=json.loads((out/'service-ready.json').read_text()),
                    actors=[json.loads((out/f'actor-{i}.json').read_text()) for i in range(4)],
                    passed=True,scope='Same-container overlapping NPU initialization and tiny BF16 matmul only; '
                    'no vLLM engine, HCCL, actor model, full memory capacity, continuous RL or recovery admission')
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');terminal=dict(exit_code=0)
        return report
    except BaseException as exc:
        terminal['error']=repr(exc);raise
    finally:
        (out/'release-service').touch()
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for stream in streams:stream.close()
        (out/'terminal.json').write_text(json.dumps(dict(terminal,finished_unix=time.time()))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--role',choices=['service','actor']);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--service-visible',default='3')
    args=p.parse_args()
    if args.role:worker(args.output,args.role,args.rank)
    else:run(args.output,args.service_visible)

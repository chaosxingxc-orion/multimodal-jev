#!/usr/bin/env python3
"""Queue resident vLLM switching after exact utility pilot acceptance."""
import argparse
import math
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read,completed,locks,launch,wait_exact
from queue_intent_finite_npu import verify_final,verify_gate


def verify_outputs(config,root):
    """Independently recompute complete reference parity, without torch."""
    root=Path(root);result=read(root/'run/result.json')
    if read(root/'run/terminal.json')['exit_code']!=0 or not result['passed']:
        raise ValueError('successful probe required')
    if result['sequence']!=['A','B','A'] or not result['a_return_exact']:
        raise ValueError('full A/B/A sequence required')
    first={};worst=0.;contrast=0.;references={}
    for key in ('A','B'):
        ref=read(Path(config['policies'][key]['gate'])/'run/vllm/result.json')
        references[key]={r['id']:r for r in ref['records']}
        if len(ref['records'])!=178 or len(references[key])!=178:
            raise ValueError('complete reference required')
    if references['A'].keys()!=references['B'].keys():raise ValueError('same reference population required')
    for uid,a in references['A'].items():
        b=references['B'][uid]
        for aa,bb in zip(a['candidate_logprobs'],b['candidate_logprobs'],strict=True):
            contrast=max(contrast,max(abs(x-y) for x,y in zip(aa,bb,strict=True)))
    if contrast<=.04:raise ValueError('distinct reference outputs required')
    for phase,key in enumerate(('A','B','A')):
        report=read(root/f'run/phase-{phase}.json');rows=report['records'];slot={'A':1,'B':2}[key]
        if (report['phase']!=phase or report['policy']!=key or report['slot']!=slot
                or report['adapter_id']!=config['policies'][key]['adapter_id']):
            raise ValueError('phase policy identity differs')
        worker=report['worker']
        if (len(worker)!=1 or worker[0]['registered_adapter_ids']!=[slot]
                or worker[0]['lora_dtype']!='torch.bfloat16'
                or worker[0]['head_projection_dtype']!='torch.bfloat16'
                or not worker[0]['actual_lora_buffer_dtypes']
                or any(v!=['torch.bfloat16'] for v in worker[0]['actual_lora_buffer_dtypes'].values())):
            raise ValueError('actual worker adapter/dtype differs')
        if len(rows)!=178 or len({r['id'] for r in rows})!=178 or {r['id'] for r in rows}!=references[key].keys():
            raise ValueError('full distinct probe population required')
        for row in rows:
            ref=references[key][row['id']]
            if row['request_id']!=ref['request_id']:raise ValueError('policy-bound request identity differs')
            aa,bb=ref['candidate_logprobs'],row['candidate_logprobs'];low=high=0.
            if not aa or len(aa)!=len(bb):raise ValueError('aligned Span support required')
            for a,b in zip(aa,bb,strict=True):
                if len(a)!=len(b) or len(a)<2 or not all(math.isfinite(x) for x in a+b):raise ValueError('finite candidate support required')
                if any(abs(math.log(sum(math.exp(x) for x in v)))>1e-6 for v in (a,b)):
                    raise ValueError('normalized candidate probability required')
                delta=[x-y for x,y in zip(a,b,strict=True)];worst=max(worst,max(map(abs,delta)))
                low+=min(delta);high+=max(delta)
            if worst>.02 or low<math.log(.98) or high>math.log(1.02):raise ValueError('independent full candidate parity failed')
            if phase==0:first[row['id']]=bb
            if phase==2 and bb!=first[row['id']]:raise ValueError('return A not exact')
    pins={str(root/'run'/name):file_hash(root/'run'/name) for name in
          ('terminal.json','result.json','runtime-profile.json','phase-0.json','phase-1.json','phase-2.json')}
    return dict(passed=True,requests_per_phase=178,sequence=['A','B','A'],maximum_logprob_error=worst,
                reference_contrast=contrast,a_return_exact=True,pins=pins,
                scope='Resident vLLM eviction/reload only; no FSDP coexistence or continuous trainer proof.')


def await_predecessor(config,out):
    prior=Path(config['predecessor']);receipt=read(prior/'launch.json')
    while True:
        check_pins(config['pins']);terminal=prior/'run/terminal.json'
        if terminal.exists():
            if read(terminal)['exit_code']!=0:raise ValueError('utility predecessor failed')
            cfg=read(prior/'config.json');protocol=read(Path(cfg['plan_root'])/'protocol.json')
            if len(protocol['arms'])!=3 or protocol['updates_per_arm']!=2:raise ValueError('full utility protocol required')
            accepted=[]
            for arm in protocol['arms']:
                if len(arm['steps'])!=2:raise ValueError('two updates required')
                for step in arm['steps']:
                    for label in ('training_root','suite_root'):
                        r=read(Path(step[label])/'launch.json')
                        if r['image']!=config['image'] or not completed(r['container_id'],config['image']):
                            raise ValueError('actual utility training/suite completion required')
                    verify_final(step['suite_root']);accepted.append(step['suite_root'])
            atomic_json(out/'predecessor-accepted.json',dict(suites=accepted));return
        try:actual=(Path('/proc')/str(receipt['pid'])/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except FileNotFoundError:
            if terminal.exists():continue
            raise ValueError('predecessor disappeared without terminal')
        if actual!=receipt['argv']:raise ValueError('predecessor PID reused')
        atomic_json(out/'progress.json',dict(stage='waiting_utility_acceptance',pid=receipt['pid'],updated_unix=time.time()))
        time.sleep(600)


def run(config,out):
    check_pins(config['pins']);root=Path(config['probe_root']);plan=read(root/'prepared.json');probe=read(root/'config.json')
    if (plan['root']!=str(root) or plan['image']!=config['image'] or plan['physical_devices']!=[4]
            or probe['sequence']!=['A','B','A']):raise ValueError('fixed probe plan required')
    check_pins({str(root/'config.json'):plan['config_sha256'],str(root/'source-manifest.json'):plan['source_manifest_sha256']})
    check_pins({str(root/'source'/p):h for p,h in read(root/'source-manifest.json').items()});check_pins(probe['pins'])
    await_predecessor(config,out)
    for key,entry in probe['policies'].items():
        gate=Path(entry['gate']);r=read(gate/'launch.json');c=read(gate/'config.json')
        if r['image']!=config['image'] or not completed(r['container_id'],config['image']):raise ValueError('reference engine actual completion required')
        if c['adapter_id']!=entry['adapter_id']:raise ValueError('reference policy differs')
        atomic_json(out/f'reference-{key}-verified.json',verify_gate(gate,config['pilot'],config['manifest_sha256']))
    with locks(Path(config['lock_directory'])):
        check_pins(config['pins']);check_pins(probe['pins'])
        receipt=launch(root,plan,plan['create_argv']);atomic_json(out/'dispatched.json',receipt)
    wait_exact(receipt['container_id'],config['image'],out,'waiting_resident_lora_switch',time.time()+600)
    atomic_json(out/'verified.json',verify_outputs(probe,root))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output);atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise

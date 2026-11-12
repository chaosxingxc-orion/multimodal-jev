#!/usr/bin/env python3
"""Queue fixed input diagnostics after the existing FSC matrix; no retries."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json, file_hash
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read, locks, launch, wait_exact
from queue_intent_prepared_acceptance import wait_resources
from score_fsc_span_transfer import authenticate_gate
from score_intent_binding_matrix import authenticate, run as score


def await_predecessor(config, output):
    root=Path(config['predecessor_controller'])
    for name in ('launch.json','config.json'):
        if config['pins'].get(str(root/name))!=file_hash(root/name): raise ValueError('exact predecessor required')
    receipt=read(root/'launch.json')
    while True:
        check_pins(config['pins']); terminal=root/'run/terminal.json'
        if terminal.exists():
            if read(terminal)['exit_code']!=0: raise ValueError('FSC predecessor failed')
            cfg=read(root/'config.json'); protocol_path=Path(cfg['matrix_root'])/'protocol.json'
            digest=file_hash(protocol_path); protocol=read(protocol_path)
            if (protocol['image']!=config['image']
                    or [e['key'] for e in protocol['checkpoints']]!=['step0']+[f'{a}-step{s}' for a in ('D','Q','P') for s in (6,12)]
                    or read(root/'run/comparison.json')['protocol_sha256']!=digest):
                raise ValueError('complete fixed predecessor matrix required')
            task=read(protocol['task']); population=read(protocol['population'])
            for entry in protocol['checkpoints']:
                authenticate_gate(entry,protocol_path,digest,protocol,population,task)
            atomic_json(output/'predecessor-accepted.json',dict(protocol_sha256=digest,
                comparison_sha256=file_hash(root/'run/comparison.json'),scope='All seven actual backend gates authenticated; no score-based selection.'))
            return
        try: actual=(Path('/proc')/str(receipt['pid'])/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except FileNotFoundError:
            if terminal.exists(): continue
            raise ValueError('predecessor missing without terminal')
        if actual!=receipt['argv']: raise ValueError('predecessor PID reused')
        atomic_json(output/'progress.json',dict(stage='waiting_FSC_matrix',pid=receipt['pid'],updated_unix=time.time()))
        time.sleep(600)


def run_matrix(config, output):
    path=Path(config['protocol']); digest=config['protocol_sha256']
    if config['pins'].get(str(path))!=digest or file_hash(path)!=digest: raise ValueError('fixed protocol required')
    protocol=read(path)
    if protocol['image']!=config['image'] or [e['key'] for e in protocol['checkpoints']]!=['step0','step4']:
        raise ValueError('fixed two-policy diagnostic required')
    for entry in protocol['checkpoints']:
        gate=Path(entry['root'])
        if any((gate/n).exists() for n in ('launch.json','create-attempt.json','run')):
            raise ValueError('already claimed; never recreate')
        for name in ('config.json','prepared.json','source-manifest.json'):
            if config['pins'].get(str(gate/name))!=file_hash(gate/name): raise ValueError('fixed gate required')
    for entry in protocol['checkpoints']:
        gate=Path(entry['root']); plan=read(gate/'prepared.json'); cfg=read(gate/'config.json')
        if (plan['image']!=config['image'] or plan['physical_devices']!=[4] or plan['root']!=str(gate)
                or plan['config_sha256']!=file_hash(gate/'config.json')
                or plan['source_manifest_sha256']!=file_hash(gate/'source-manifest.json')
                or cfg['pins'].get(str(path))!=digest): raise ValueError('gate identity differs')
        stage=output/entry['key']; stage.mkdir(exist_ok=False)
        wait_resources(dict(image=config['image'],pins=config['pins']),stage)
        with locks(Path(config['lock_directory'])):
            check_pins(config['pins']); check_pins(cfg['pins'])
            check_pins({str(gate/'source'/p):h for p,h in read(gate/'source-manifest.json').items()})
            receipt=launch(gate,plan,plan['create_argv']); atomic_json(stage/'dispatched.json',receipt)
        wait_exact(receipt['container_id'],config['image'],stage,'waiting_binding_policy',time.time()+600)
        _,audit=authenticate(entry,path,digest,protocol); atomic_json(stage/'backend-accepted.json',audit)
        atomic_json(output/'progress.json',dict(stage='accepted_binding_member',key=entry['key'],updated_unix=time.time()))
    atomic_json(output/'comparison.json',score(path,digest))


def run(config, output):
    check_pins(config['pins']); await_predecessor(config,output); run_matrix(config,output)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output); atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time())); raise

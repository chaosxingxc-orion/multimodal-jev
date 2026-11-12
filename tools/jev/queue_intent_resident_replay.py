#!/usr/bin/env python3
"""Two-step resident actor replay after accepted resident vLLM switch probe."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.container_pins import validate_bind_pins
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read,completed,locks,launch,wait_exact
from queue_span_lora_switch import verify_outputs
from queue_intent_finite_npu import verify_final
from probe_intent_resident_replay_npu import validate


def await_predecessor(config,out):
    root=Path(config['predecessor']);receipt=read(root/'launch.json')
    for name in ('config.json','launch.json'):
        if config['pins'].get(str(root/name))!=file_hash(root/name):raise ValueError('pinned predecessor required')
    while True:
        check_pins(config['pins']);terminal=root/'run/terminal.json'
        if terminal.exists():
            if read(terminal)['exit_code']!=0:raise ValueError('resident switch predecessor failed')
            prior=read(root/'config.json');probe=Path(prior['probe_root']);dispatched=read(root/'run/dispatched.json')
            if (dispatched!=read(probe/'launch.json') or dispatched['image']!=config['image']
                    or not completed(dispatched['container_id'],config['image'])):
                raise ValueError('actual switch completion required')
            check_pins(read(probe/'config.json')['pins'])
            atomic_json(out/'predecessor-verified.json',verify_outputs(read(probe/'config.json'),probe));return
        try:actual=(Path('/proc')/str(receipt['pid'])/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except FileNotFoundError:
            if terminal.exists():continue
            raise ValueError('predecessor missing without terminal')
        if actual!=receipt['argv']:raise ValueError('predecessor PID reused')
        atomic_json(out/'progress.json',dict(stage='waiting_resident_switch',pid=receipt['pid'],updated_unix=time.time()))
        time.sleep(600)


def freeze_gate(training,suite):
    training,suite=map(Path,(training,suite));plan=read(suite/'prepared.json')
    if any((suite/n).exists() for n in ('config.json','run','launch.json','create-attempt.json')):raise ValueError('suite already claimed')
    check_pins({str(suite/'config-template.json'):plan['config_template_sha256'],str(suite/'source-manifest.json'):plan['source_manifest_sha256']})
    cfg=read(suite/'config-template.json');check_pins(cfg['pins'])
    check_pins({str(suite/'source'/p):h for p,h in read(suite/'source-manifest.json').items()})
    if cfg['resident_training_root']!=str(training) or cfg['adapter']!=str(training/'run/resident-adapter'):raise ValueError('suite replay lineage differs')
    adapter_id,files=checkpoint_identity(cfg['adapter'])
    if adapter_id!=cfg['adapter_id']:raise ValueError('replayed final adapter differs')
    for rank in range(4):
        first=training/'run'/f'rank-{rank}';second=training/'run'/f'resident-rank-{rank}'
        if read(first/'terminal.json')['exit_code']!=0:raise ValueError('all resident ranks required')
        result=read(second/'result.json')
        if (result['passed'] is not True or result['rank']!=rank or result['updates']!=2
                or result['phase']!='awaiting_acceptance' or result['resident_objects_preserved'] is not True
                or result['policy_id']!=cfg['base_id']+':'+adapter_id):raise ValueError('complete second-update evidence required')
    for p in (training/'run').rglob('*'):
        if p.is_file() and p.suffix in ('.json','.pt','.safetensors'):cfg['pins'][str(p)]=file_hash(p)
    for p in (training/'launch.json',training/'config.json',training/'source-manifest.json'):cfg['pins'][str(p)]=file_hash(p)
    atomic_json(suite/'bind-pins.json',validate_bind_pins(plan['create_argv'],cfg['pins']))
    atomic_json(suite/'config.json',cfg)
    return plan


def verify_accepted(suite):
    suite=Path(suite);accepted=read(suite/'ACCEPTED.json');verified=read(suite/'verified.json')
    if (read(suite/'controller-terminal.json')['exit_code']!=0 or accepted['passed'] is not True
            or accepted['verified_sha256']!=file_hash(suite/'verified.json') or verified['passed'] is not True
            or verified['updates']!=2 or len(verified['ranks'])!=4 or verified['full_gate']['requests']!=178
            or accepted['adapter_id']!=verified['adapter_id']):raise ValueError('independent resident acceptance required')
    check_pins(verified['files']);return accepted


def run(config,out):
    check_pins(config['pins']);training=Path(config['training']);suite=Path(config['suite']);plan=read(training/'prepared.json')
    if plan['root']!=str(training) or plan['image']!=config['image'] or plan['physical_devices']!=[4,5,6,7]:raise ValueError('fixed replay plan required')
    check_pins({str(training/'config.json'):plan['config_sha256'],str(training/'source-manifest.json'):plan['source_manifest_sha256']})
    check_pins({str(training/'source'/p):h for p,h in read(training/'source-manifest.json').items()});probe=read(training/'config.json');validate(probe)
    await_predecessor(config,out)
    for entry in probe['references']:
        for name in ('training','suite'):
            r=read(Path(entry[name])/'launch.json')
            if r['image']!=config['image'] or not completed(r['container_id'],config['image']):raise ValueError('actual reference completion required')
        verify_final(entry['suite'])
    with locks(Path(config['lock_directory'])):
        check_pins(config['pins']);receipt=launch(training,plan,plan['create_argv']);atomic_json(out/'training-dispatched.json',receipt)
    wait_exact(receipt['container_id'],config['image'],out,'waiting_resident_actor',time.time()+600)
    with locks(Path(config['lock_directory'])):
        check_pins(config['pins']);sp=freeze_gate(training,suite)
        if sp['image']!=config['image'] or sp['physical_devices']!=[4]:raise ValueError('fixed TP1 suite required')
        receipt=launch(suite,sp,sp['create_argv']);atomic_json(out/'suite-dispatched.json',receipt)
    wait_exact(receipt['container_id'],config['image'],out,'waiting_resident_acceptance',time.time()+600)
    atomic_json(out/'accepted.json',verify_accepted(suite))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output);atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise

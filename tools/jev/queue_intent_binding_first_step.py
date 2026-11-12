#!/usr/bin/env python3
"""One binding update after the full existing queue; no recreation or retries."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_binding_resume import validate_config,expected_selection
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read,completed,locks,launch,wait_exact
from queue_intent_prepared_acceptance import wait_resources
from freeze_intent_binding_training import freeze as freeze_training
from freeze_intent_finite_acceptance import freeze as freeze_suite
from run_intent_momentum_continuation import validate as validate_momentum,authenticate_pair


def validate(config):
    check_pins(config['pins'])
    root=Path(config['plan_root']);protocol=read(root/'protocol.json')
    if config['pins'].get(str(root/'protocol.json'))!=file_hash(root/'protocol.json'):
        raise ValueError('pinned binding protocol required')
    if (protocol['schema']!='intent-binding-first-update-protocol-v1' or protocol['arm']!='D'
            or protocol['updates']!=1 or protocol['source_uses']!=8 or protocol['view_uses']!=32
            or protocol['initial_policy_id']!=config['policy_id']):
        raise ValueError('fixed first D binding update required')
    for kind in ('training','suite'):
        out=root/kind
        for name in ('prepared.json','config-template.json','source-manifest.json'):
            if config['pins'].get(str(out/name))!=file_hash(out/name):raise ValueError('complete fixed plan required')
        plan=read(out/'prepared.json');cfg=read(out/'config-template.json')
        if (plan['root']!=str(out) or plan['image']!=config['image']
                or plan['physical_devices']!=([4,5,6,7] if kind=='training' else [4])
                or plan['config_template_sha256']!=file_hash(out/'config-template.json')
                or plan['source_manifest_sha256']!=file_hash(out/'source-manifest.json')):
            raise ValueError('binding plan identity differs')
        check_pins(cfg['pins']);check_pins({str(out/'source'/p):h for p,h in read(out/'source-manifest.json').items()})
    cfg=read(root/'training/config-template.json');validate_config(cfg)
    _,selection=expected_selection(cfg)
    suite=read(root/'suite/config-template.json')
    if (cfg['step']!=0 or cfg['arm']!='D' or selection!=protocol['selection']
            or suite['training_root']!=str(root/'training') or suite['adapter']!=str(root/'training/run/adapter')
            or suite['population']!=dict(recordings=89,requests=712)):
        raise ValueError('binding schedule or acceptance pairing differs')


def await_predecessor(config,output):
    root=Path(config['predecessor_controller'])
    for name in ('config.json','launch.json'):
        if config['pins'].get(str(root/name))!=file_hash(root/name):raise ValueError('exact momentum predecessor required')
    receipt=read(root/'launch.json')
    while True:
        check_pins(config['pins']);terminal=root/'run/terminal.json'
        if terminal.exists():
            if read(terminal)['exit_code']!=0:raise ValueError('momentum predecessor failed')
            predecessor=read(root/'config.json');validate_momentum(predecessor)
            progress=read(root/'run/progress.json')
            if (predecessor['image']!=config['image'] or progress['completed_updates']!=12
                    or progress['total_updates']!=12 or progress['accepted_suite']!=predecessor['steps'][-1]['suite_root']):
                raise ValueError('full momentum continuation required')
            steps=[dict(training_root=predecessor['previous_training'],suite_root=predecessor['previous_suite']),*predecessor['steps']]
            if len(steps)!=12:raise ValueError('all twelve predecessor updates required')
            receipts=[authenticate_pair(step,config['image'],i+1) for i,step in enumerate(steps)]
            atomic_json(output/'predecessor-accepted.json',dict(updates=12,receipts=receipts))
            return
        try:actual=(Path('/proc')/str(receipt['pid'])/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except FileNotFoundError:
            if terminal.exists():continue
            raise ValueError('predecessor missing without terminal')
        if actual!=receipt['argv']:raise ValueError('predecessor PID reused')
        atomic_json(output/'progress.json',dict(stage='waiting_full_beta1zero_continuation',pid=receipt['pid'],updated_unix=time.time()))
        time.sleep(600)


def verify_final(root,training,image):
    root,training=Path(root),Path(training)
    for path in (training,root):
        receipt=read(path/'launch.json')
        if receipt['image']!=image or not completed(receipt['container_id'],image):
            raise ValueError('actual training and suite completion required')
        check_pins({str(path/'config.json'):receipt['config_sha256'],str(path/'source-manifest.json'):receipt['source_manifest_sha256']})
        check_pins(read(path/'config.json')['pins'])
        check_pins({str(path/'source'/p):h for p,h in read(path/'source-manifest.json').items()})
    if read(root/'controller-terminal.json')['exit_code']!=0:raise ValueError('binding acceptance failed')
    accepted=read(root/'ACCEPTED.json');verified=read(root/'verified.json')
    cfg=read(training/'config.json');suite=read(root/'config.json')
    if (accepted.get('passed') is not True or verified.get('passed') is not True
            or accepted['verified_sha256']!=file_hash(root/'verified.json')
            or verified['schema']!='intent-binding-update-verified-v1' or verified['updates']!=1
            or verified['arm']!='D' or verified['root']!=str(training) or verified['suite']!=str(root)
            or verified['source_uses']!=8 or verified['view_uses']!=32 or verified['full_gate_requests']!=712
            or verified['binding_manifest_sha256']!=cfg['binding_manifest_sha256']
            or verified.get('exact_export_reload') is not True
            or accepted['adapter_id']!=verified['adapter_id'] or suite['adapter_id']!=verified['adapter_id']):
        raise ValueError('complete binding update acceptance required')
    required={str(training/'run'/n) for n in ('selection.json','supervision.json')}
    required.update(str(training/'run'/f'rank-{r}'/n) for r in range(4) for n in
        ('optimizer-local.pt','result.json','terminal.json','selection.json','before.json','after.json'))
    if not required<=verified['files'].keys():raise ValueError('complete binding rank evidence required')
    check_pins(verified['files'])
    return accepted


def run(config,output):
    validate(config);root=Path(config['plan_root']);training=root/'training';suite=root/'suite'
    if any((p/n).exists() for p in (training,suite) for n in ('config.json','launch.json','create-attempt.json','run')):
        raise ValueError('plan already claimed; inspect existing task')
    await_predecessor(config,output)
    wait_resources(config,output)
    with locks(Path(config['lock_directory'])):
        validate(config)
        audit=freeze_training(training,config['policy_id']);atomic_json(output/'input-gate-accepted.json',audit)
        plan=read(training/'prepared.json');receipt=launch(training,plan,plan['create_argv'])
        atomic_json(output/'training-dispatched.json',receipt)
    wait_exact(receipt['container_id'],config['image'],output,'waiting_binding_update',time.time()+600)
    wait_resources(config,output)
    with locks(Path(config['lock_directory'])):
        check_pins(config['pins']);freeze_suite(suite)
        plan=read(suite/'prepared.json');receipt=launch(suite,plan,plan['create_argv'])
        atomic_json(output/'acceptance-dispatched.json',receipt)
    wait_exact(receipt['container_id'],config['image'],output,'waiting_binding_acceptance',time.time()+600)
    atomic_json(output/'accepted.json',verify_final(suite,training,config['image']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output);atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise

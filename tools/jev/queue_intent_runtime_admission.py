#!/usr/bin/env python3
"""Wait an exact native update, verify export independently, then gate D/Q/P."""
import argparse
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.container_pins import validate_bind_pins
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import wait_exact,freeze_suite,locks,launch,read


def verify_suite_terminal(root):
    if read(root/'controller-terminal.json')['exit_code']!=0:raise ValueError('native acceptance controller failed')
    accepted=read(root/'ACCEPTED.json');verified=read(root/'verified.json')
    if not accepted['passed'] or accepted['verified_sha256']!=file_hash(root/'verified.json'):
        raise ValueError('native acceptance changed')
    if not verified['native_actor_verified'] or not verified['matched_bf16_export_exact']:
        raise ValueError('native actor equivalence not verified')
    check_pins(verified['files']);check_pins(read(root/'config.json')['pins'])
    return accepted


def run(config,output):
    check_pins(config['pins'])
    parent=read(Path(config['native_root'])/'launch.json')
    if parent['container_id']!=config['native_container']:raise ValueError('native container differs')
    wait_exact(parent['container_id'],config['image'],output,'waiting_exact_native_update',config['first_poll_after'])
    with locks(Path(config['lock_directory'])):
        check_pins(config['pins']);suite_plan=freeze_suite(config);suite=Path(config['suite_root'])
        if suite_plan['image']!=config['image'] or suite_plan['physical_devices']!=[4]:raise ValueError('suite topology differs')
        audit=validate_bind_pins(suite_plan['create_argv'],read(suite/'config.json')['pins']);atomic_json(suite/'bind-pins.json',audit)
        receipt=launch(suite,suite_plan,suite_plan['create_argv']);atomic_json(output/'acceptance-dispatched.json',receipt)
    wait_exact(receipt['container_id'],config['image'],output,'waiting_native_acceptance',time.time()+600)
    accepted=verify_suite_terminal(suite);atomic_json(output/'native-accepted.json',accepted)
    with locks(Path(config['lock_directory'])):
        check_pins(config['pins']);plan=read(config['intent_plan']);root=Path(plan['root'])
        if plan['status']!='prepared_not_launched' or plan['physical_devices']!=[4] or plan['image']!=config['image']:
            raise ValueError('intent gate plan differs')
        check_pins({str(root/'config.json'):plan['config_sha256'],str(root/'source-manifest.json'):plan['source_manifest_sha256']})
        cfg=read(root/'config.json');check_pins(cfg['pins'])
        check_pins({str(root/'source'/p):h for p,h in read(root/'source-manifest.json').items()})
        audit=validate_bind_pins(plan['create_argv'],cfg['pins']);atomic_json(root/'bind-pins.json',audit)
        receipt=launch(root,plan,plan['create_argv']);atomic_json(output/'intent-gate-dispatched.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output)
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time(),scope='Native accepted and D/Q/P gate dispatched; D/Q/P not yet accepted.'))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise

#!/usr/bin/env python3
"""One synthetic FSDP residency probe after the accepted momentum first step."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json, file_hash
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read, completed, locks, launch, wait_exact
from queue_intent_finite_npu import verify_final
from queue_intent_prepared_acceptance import wait_resources


def await_predecessor(config, output):
    root=Path(config['predecessor_controller'])
    for name in ('launch.json','config.json'):
        if config['pins'].get(str(root/name))!=file_hash(root/name): raise ValueError('exact predecessor required')
    receipt=read(root/'launch.json')
    while True:
        check_pins(config['pins']);terminal=root/'run/terminal.json'
        if terminal.exists():
            if read(terminal)['exit_code']!=0: raise ValueError('momentum predecessor failed')
            progress=read(root/'run/progress.json');suite=Path(config['predecessor_suite'])
            if progress.get('completed_updates')!=1 or progress.get('accepted_suite')!=str(suite):
                raise ValueError('complete first update required')
            for target in (suite,Path(read(suite/'config.json')['training_root'])):
                launched=read(target/'launch.json')
                if launched['image']!=config['image'] or not completed(launched['container_id'],config['image']):
                    raise ValueError('actual successful predecessor required')
            accepted=verify_final(suite);verified=read(suite/'verified.json')
            if verified['arm']!='D' or verified['updates']!=1: raise ValueError('momentum first-step identity differs')
            atomic_json(output/'predecessor-accepted.json',accepted);return
        try:actual=(Path('/proc')/str(receipt['pid'])/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except FileNotFoundError:
            if terminal.exists():continue
            raise ValueError('predecessor missing without terminal')
        if actual!=receipt['argv']:raise ValueError('predecessor PID reused')
        atomic_json(output/'progress.json',dict(stage='waiting_momentum_first_acceptance',pid=receipt['pid'],updated_unix=time.time()))
        time.sleep(600)


def verify_results(root):
    root=Path(root);ranks=[];hashes=None
    for rank in range(4):
        out=root/'run'/f'rank-{rank}';result=read(out/'result.json')
        if (read(out/'terminal.json')['exit_code']!=0 or result['status']!='passed'
                or result['rank']!=rank or result['world_size']!=4 or result['updates']!=3
                or result['restored_at_update']!=2 or result['backend']!='hccl'):
            raise ValueError('complete four-rank synthetic probe required')
        for field in ('exact_resume_output','exact_resume_masters','exact_resume_adam','frozen_base_unchanged','cross_rank_export_exact'):
            if result[field] is not True:raise ValueError('probe invariant failed: '+field)
        residency=result['residency_roundtrip']
        for field in ('exact_parameters_and_adam','original_tensor_devices_preserved','parameter_references_preserved'):
            if residency[field] is not True:raise ValueError('residency invariant failed: '+field)
        if result['execution_dtypes']!=[['torch.bfloat16','torch.bfloat16']]:raise ValueError('BF16 execution required')
        if hashes is not None and hashes!=result['adapter_export_hashes']:raise ValueError('rank exports differ')
        hashes=result['adapter_export_hashes']
        if file_hash(out/'local-checkpoint.pt')!=result['checkpoint_sha256']:raise ValueError('checkpoint changed')
        ranks.append(dict(rank=rank,result_sha256=file_hash(out/'result.json'),residency=residency))
    return dict(passed=True,ranks=ranks,scope='Synthetic FSDP offload/reload and next update; no audio/vLLM capacity or efficacy proof.')


def run(config,output):
    check_pins(config['pins']);root=Path(config['probe_root']);plan=read(root/'prepared.json')
    for name in ('prepared.json','config.json','source-manifest.json'):
        if config['pins'].get(str(root/name))!=file_hash(root/name):raise ValueError('fixed probe required')
    if (plan['image']!=config['image'] or plan['physical_devices']!=[4,5,6,7]
            or plan['root']!=str(root) or '--residency-roundtrip' not in plan['create_argv']):raise ValueError('fixed residency probe required')
    await_predecessor(config,output);wait_resources(config,output)
    with locks(Path(config['lock_directory'])):
        check_pins(config['pins']);check_pins(read(root/'config.json')['pins'])
        receipt=launch(root,plan,plan['create_argv']);atomic_json(output/'dispatched.json',receipt)
    wait_exact(receipt['container_id'],config['image'],output,'waiting_residency_probe',time.time()+600)
    check_pins(config['pins']);check_pins(read(root/'config.json')['pins'])
    result=verify_results(root);atomic_json(root/'verified.json',result);atomic_json(output/'verified.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output);atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise

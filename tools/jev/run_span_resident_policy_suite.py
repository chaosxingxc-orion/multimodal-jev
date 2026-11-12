#!/usr/bin/env python3
"""Fresh HF references followed by one resident vLLM over fixed policy exports.

No concurrent HF/vLLM execution or actor co-residency. This is a gate producer,
not an optimizer loop; host must authenticate actual container exit separately.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.vllm_spans import register_span_model
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read

# EngineCore spawn must register the custom architecture before loading it.
register_span_model()


def validate(config, output):
    check_pins(config['pins'])
    if config['schema'] != 'jev-resident-policy-suite-v1' or len(config['policies']) < 2:
        raise ValueError('registered multi-policy suite required')
    configs=[];seen=set();runtime=None
    for entry in config['policies']:
        key=entry['key']
        if (not isinstance(key,str) or not key or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in key)
                or key in seen):
            raise ValueError('distinct safe policy keys required')
        seen.add(key);path=Path(entry['config'])
        if config['pins'].get(str(path)) != file_hash(path):
            raise ValueError('pinned policy configuration required')
        c=read(path);check_pins(c['pins'])
        if (c['physical_devices'] != [4] or c['max_model_len'] != 4096
                or checkpoint_identity(c['model'])[0] != c['base_id']
                or checkpoint_identity(c['adapter'])[0] != c['adapter_id']):
            raise ValueError('fixed TP1 runtime and immutable policy required')
        identity={k:c[k] for k in ('model','base_id','runtime_profile','max_model_len','inputs','population')}
        if runtime is None:runtime=identity
        elif runtime != identity:raise ValueError('same runtime and input population required')
        artifact=read(c['inputs']);count=c['population']['requests']
        if (c['pins'].get(c['inputs']) != file_hash(c['inputs']) or artifact.get('passed') is not True
                or artifact['requests'] != count or len(artifact['records']) != count
                or len({r['id'] for r in artifact['records']}) != count
                or artifact['recordings'] != c['population']['recordings']):
            raise ValueError('complete pinned processor population required')
        if (output/key).exists():raise ValueError('policy output already claimed; never overwrite')
        configs.append((key,c,artifact))
    return configs


def verify_pair(config, artifact, hf, vl):
    """Recompute parity from outputs; do not trust worker passed/parity flags."""
    from multimodal_jev.span_parity import compare_span_policies
    count=config['population']['requests'];expected={r['id'] for r in artifact['records']}
    for backend,report in (('hf',hf),('vllm',vl)):
        if (report.get('passed') is not True or report['backend'] != backend
                or report['base_id'] != config['base_id'] or report['adapter_id'] != config['adapter_id']
                or len(report['records']) != count or {r['id'] for r in report['records']} != expected):
            raise ValueError('complete same-policy independent outputs required')
    refs={r['id']:r for r in hf['records']};worst=0.
    for row in vl['records']:
        ref=refs[row['id']]
        if row['request_id'] != ref['request_id'] or row['request_identity'] != ref['request_identity']:
            raise ValueError('exact reference request required')
        measured=compare_span_policies(ref['candidate_logprobs'],row['candidate_logprobs'])
        if not measured['passed']:raise ValueError('independent resident parity failed')
        worst=max(worst,measured['maximum_logprob_error'])
    return dict(requests=count,maximum_logprob_error=worst)


def run_hf(config, path, output):
    subprocess.run([sys.executable,str(Path(__file__).with_name('check_slurp_native_span_policy.py')),
        '--config',str(path),'--output',str(output),'--phase','hf'],check=True)
    if read(output/'terminal.json')['exit_code'] != 0:raise ValueError('HF phase failed')
    return read(output/'result.json')


def run(config, output):
    from check_slurp_native_span_policy import prepared
    from multimodal_jev.span_resident_rollout import ResidentSpanRollout
    configs=validate(config,output);references={};snapshots={}
    # Every HF subprocess exits before creating the resident engine. This avoids
    # assuming that HF and vLLM can safely share the same NPU allocation.
    for key,c,artifact in configs:
        root=output/key;root.mkdir();atomic_json(root/'config.json',c);(root/'run').mkdir()
        references[key]=run_hf(c,root/'config.json',root/'run/hf')
        for name in ('result.json','terminal.json'):
            snapshots[str(root/'run/hf'/name)]=file_hash(root/'run/hf'/name)
        atomic_json(output/'progress.json',dict(stage='hf_reference_completed',policy=key,updated_unix=time.time()))
    check_pins(config['pins']);rollout=ResidentSpanRollout(configs[0][1],configs[0][1]['adapter'])
    audits=[]
    try:
        atomic_json(output/'runtime-profile.json',rollout.runtime_receipt)
        for key,c,artifact in configs:
            root=output/key;vlroot=root/'run/vllm';vlroot.mkdir()
            check_pins(config['pins']);check_pins(c['pins']);check_pins(snapshots)
            def progress(done,total):
                atomic_json(output/'progress.json',dict(stage='resident_vllm',policy=key,completed=done,total=total,updated_unix=time.time()))
            report=rollout.evaluate(c,prepared(c,artifact),references[key],progress)
            audit=verify_pair(c,artifact,references[key],report)
            atomic_json(vlroot/'result.json',report)
            atomic_json(vlroot/'worker.json',report['worker'])
            atomic_json(vlroot/'runtime-profile.json',rollout.runtime_receipt)
            atomic_json(vlroot/'terminal.json',dict(exit_code=0,scope='Completed policy phase inside resident engine; not a separate process exit.'))
            audits.append(dict(key=key,adapter_id=c['adapter_id'],slot=report['slot'],**audit))
    finally:
        rollout.close()
    # Publish READY only after the engine shuts down successfully and every
    # policy passed. Outer host authentication of actual exit remains required.
    check_pins(snapshots);check_pins(config['pins']);pins=dict(snapshots)
    for key,c,artifact in configs:
        root=output/key
        atomic_json(root/'run/READY.json',dict(passed=True,requests=c['population']['requests'],base_id=c['base_id'],
            adapter_id=c['adapter_id'],result_sha256=file_hash(root/'run/vllm/result.json'),
            scope='Resident inference parity only; no actor state or training admission.'))
        atomic_json(root/'run/terminal.json',dict(exit_code=0,finished_unix=time.time()))
        for name in ('config.json','run/READY.json','run/terminal.json','run/vllm/result.json','run/vllm/terminal.json'):
            pins[str(root/name)]=file_hash(root/name)
    return dict(passed=True,policies=audits,files=pins,
        scope='Fresh independent HF per policy, then one resident vLLM. No actor updates, co-residency or efficacy claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'result.json',run(read(a.config),a.output))
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise

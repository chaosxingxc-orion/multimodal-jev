#!/usr/bin/env python3
"""Candidate resident vLLM switch step2→step4→step2 over full256 inputs.

Requires independently accepted candidate instruction policy gates. This tests one
engine's version switching, not a fresh FSDP restore or a second RL update.
"""
import argparse,json,time
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
# Spawned vLLM workers must register the Span architecture on import.
from multimodal_jev.vllm_spans import register_span_model
register_span_model()


def run(config,output):
    from verify_slurp_batched_parity import verify
    from check_span_candidate_reload import prepared,validate
    from multimodal_jev.span_batched_resident_rollout import BatchedResidentSpanRollout
    from multimodal_jev.span_batch_training_config import PROFILE
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('switch dependency changed: '+path)
    audits=[Path(config[k]) for k in ('old_audit_config','updated_audit_config')]
    if any(config['pins'].get(str(p))!=file_hash(p) for p in audits):
        raise ValueError('pinned independent candidate audit configs required')
    accepted=[verify(p) for p in audits]
    if (accepted[0]['updates']!=2 or accepted[1]['updates']!=4
            or any(r['requests']!=256 or r['population_mode']!='prefix' for r in accepted)
            or accepted[0]['adapter_id']==accepted[1]['adapter_id']):
        raise ValueError('distinct accepted candidate step2 and step4 policies required')
    gates=[Path(json.loads(p.read_text())['backend_root']) for p in audits]
    policies=[json.loads((p/'config.json').read_text()) for p in gates]
    for field in ('model','base_id','runtime_profile','max_model_len','inputs','population'):
        if policies[0][field]!=policies[1][field]:raise ValueError('switch population/runtime differs: '+field)
    for policy in policies:
        policy['physical_devices']=config['physical_devices']
        policy['candidate_profile']=PROFILE
    artifacts=[validate(p) for p in policies]
    references=[json.loads((g/'run/hf/result.json').read_text()) for g in gates]
    engine=BatchedResidentSpanRollout(policies[0],policies[0]['adapter'])
    atomic_json(output/'runtime-profile.json',engine.runtime_receipt)
    results=[]
    try:
        for phase,index in enumerate((0,1,0)):
            start=time.monotonic()
            result=engine.evaluate(policies[index],prepared(policies[index],artifacts[index]),references[index],
                progress=lambda done,total:atomic_json(output/'progress.json',dict(phase=phase,completed=done,total=total)))
            atomic_json(output/f'policy-{phase}.json',result)
            results.append(result)
            atomic_json(output/f'timing-{phase}.json',dict(seconds=time.monotonic()-start))
        if results[0]['records']!=results[2]['records']:
            raise ValueError('switch back did not restore exact original output')
        # Same resident engine, now collect a fresh behavior without supplying
        # HF outputs to the execution interface. Audit it independently below.
        behavior=engine.rollout(policies[1],prepared(policies[1],artifacts[1]))
        atomic_json(output/'behavior.json',behavior)
        from verify_simple_sampled_switch import compare_behavior
        behavior_check=compare_behavior(behavior,results[1])
        files={str(p):file_hash(p) for p in output.glob('policy-*.json')}
        files[str(output/'behavior.json')]=file_hash(output/'behavior.json')
        return dict(passed=True,requests_per_phase=256,phases=3,
            adapter_sequence=[policies[i]['adapter_id'] for i in (0,1,0)],candidate_profile=PROFILE,
            exact_switch_back=True,files=files,reference_free_behavior=behavior_check,
            scope='Candidate single resident engine step2/step4/step2 full256 with independent HF references. No RL update, fresh FSDP restore or continuous RL efficacy.')
    finally:engine.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'result.json',run(json.loads(a.config.read_text()),a.output))
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise

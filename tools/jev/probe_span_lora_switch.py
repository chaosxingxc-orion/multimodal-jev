#!/usr/bin/env python3
"""One resident TP1 vLLM engine, accepted A/B/A adapters, full audio population."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.policy_identity import checkpoint_identity
from queue_span_native_actor_npu import read
from multimodal_jev.vllm_spans import register_span_model

# Spawned EngineCore imports this module before resolving the architecture.
register_span_model()


def validate(config):
    from queue_span_dp4_resume import check_pins
    check_pins(config['pins'])
    if (config['sequence']!=['A','B','A'] or set(config['policies'])!={'A','B'}
            or config['max_model_len']!=4096 or config['physical_devices']!=[4]):
        raise ValueError('fixed A/B/A TP1 population required')
    configs={};records={}
    for key,entry in config['policies'].items():
        root=Path(entry['gate']);c=read(root/'config.json');vl=read(root/'run/vllm/result.json')
        if (entry['adapter_id']!=c['adapter_id'] or checkpoint_identity(c['adapter'])[0]!=entry['adapter_id']
                or c['model']!=config['model'] or c['base_id']!=config['base_id']
                or c['runtime_profile']!=config['runtime_profile'] or not vl['passed']
                or vl['adapter_id']!=entry['adapter_id'] or vl['base_id']!=config['base_id']):
            raise ValueError('accepted immutable policy differs')
        for name in ('config.json','run/vllm/result.json','run/hf/result.json','run/READY.json'):
            if config['pins'].get(str(root/name))!=file_hash(root/name):raise ValueError('complete pinned policy evidence required')
        if c['population']!={'recordings':89,'requests':178}:raise ValueError('full original pilot required')
        configs[key]=c;records[key]={r['id']:r for r in vl['records']}
        if len(records[key])!=178 or len(vl['records'])!=178:raise ValueError('complete distinct reference population required')
    if (configs['A']['inputs']!=configs['B']['inputs'] or records['A'].keys()!=records['B'].keys()
            or configs['A']['adapter_id']==configs['B']['adapter_id']):
        raise ValueError('same inputs and distinct adapters required')
    maximum=max(abs(a-b) for uid in records['A'] for xs,ys in zip(records['A'][uid]['candidate_logprobs'],records['B'][uid]['candidate_logprobs'],strict=True) for a,b in zip(xs,ys,strict=True))
    if maximum<=.04:raise ValueError('reference policies insufficiently distinct to detect stale output')
    return configs,records,maximum


def run(config,out):
    import torch
    import torch_npu
    from multimodal_jev.span_resident_rollout import ResidentSpanRollout
    from check_slurp_native_span_policy import prepared,validate as validate_population
    configs,references,contrast=validate(config)
    torch.set_num_threads(4);torch.npu.set_device(0)
    rollout=ResidentSpanRollout(config,configs['A']['adapter'])
    reports=[];first={};started=time.monotonic()
    try:
        atomic_json(out/'runtime-profile.json',rollout.runtime_receipt)
        for phase,key in enumerate(config['sequence']):
            c=configs[key];artifact=validate_population(c)
            reference=read(Path(config['policies'][key]['gate'])/'run/vllm/result.json')
            def progress(done,total):
                atomic_json(out/'progress.json',dict(phase=phase,policy=key,completed=done,total=total,updated_unix=time.time()))
            report=rollout.evaluate(c,prepared(c,artifact),reference,progress)
            if report['slot']!={'A':1,'B':2}[key]:raise ValueError('A/B/A adapter registration differs')
            for row in report['records']:
                if phase==0:first[row['id']]=row['candidate_logprobs']
                if phase==2 and row['candidate_logprobs']!=first[row['id']]:raise ValueError('A return probabilities not exact')
            report.update(phase=phase,policy=key)
            atomic_json(out/f'phase-{phase}.json',report)
            reports.append(dict(phase=phase,policy=key,requests=len(report['records']),
                maximum_logprob_error=max(r['parity']['maximum_logprob_error'] for r in report['records'])))
        return dict(passed=True,sequence=config['sequence'],reports=reports,reference_policy_contrast=contrast,
            a_return_exact=True,seconds=time.monotonic()-started,
            scope='One resident vLLM TP1 engine, A/B/A LoRA eviction/reload and178 actual audio inputs per phase. No FSDP actor co-residency, optimizer update or full verl trainer claim.')
    finally:rollout.close()


if __name__=='__main__':
    # EngineCore spawn needs architecture registration before model loading.
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'result.json',run(read(a.config),a.output));atomic_json(a.output/'terminal.json',{'exit_code':0})
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise

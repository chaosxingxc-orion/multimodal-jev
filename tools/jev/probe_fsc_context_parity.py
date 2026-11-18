#!/usr/bin/env python3
"""Candidate HF/vLLM parity on a label-free deterministic FSC input panel.

Separate processes for HF and vLLM. No target labels, task scores or source
policy selection. Run only through a separately frozen device launcher.
"""
import argparse
import json
import math
from pathlib import Path

from multimodal_jev.baseline import file_hash
from infer_fsc_context_transfer import load_inputs,audited_encoder,checked_prediction


def panel_indices(rows,receipts):
    if len(rows)!=len(receipts) or not rows:
        raise ValueError('aligned input population required')
    speakers={}
    for i,row in enumerate(rows):speakers.setdefault(row['speaker_id'],i)
    chosen=set(speakers.values())
    ordered=sorted(range(len(rows)),key=lambda i:(receipts[i]['tokens'],rows[i]['id']))
    chosen.update(ordered[:16]);chosen.update(ordered[-16:])
    return sorted(chosen)


def compare(hf,vl):
    if (hf['policy_id']!=vl['policy_id'] or hf['indices']!=vl['indices'] or
            hf['input_manifest_sha256']!=vl['input_manifest_sha256'] or
            len(hf['records'])!=len(vl['records']) or not hf['records'] or
            len(hf['records'])!=len(hf['indices']) or
            len(set(hf['indices']))!=len(hf['indices']) or
            len({row['id'] for row in hf['records']})!=len(hf['records'])):
        raise ValueError('same policy and nonempty input panel required')
    maximum=0.;flips=0
    for a,b in zip(hf['records'],vl['records'],strict=True):
        if a['id']!=b['id'] or a['request_identity']!=b['request_identity']:
            raise ValueError('backend inputs differ')
        if len(a['prob'])!=31 or len(b['prob'])!=31:
            raise ValueError('complete native31 probabilities required')
        for x,y in zip(a['prob'],b['prob'],strict=True):
            if any(type(p) not in (int,float) or not math.isfinite(p) or not 0<=p<=1 for p in (x,y)):
                raise ValueError('finite probabilities required')
            maximum=max(maximum,abs(x-y));flips+=int((x>=.5)!=(y>=.5))
    return dict(queries=len(hf['records']),decisions=31,probability_max_abs=maximum,
                threshold_flips=flips,exact=maximum==0 and flips==0,
                scope='Fixed input panel probability comparison; full transfer and independent provenance audit separate')


def run(config,out,phase):
    import torch
    import torch_npu
    from multimodal_jev.local_checkpoint import commit_json
    from multimodal_jev.local_runtime_storage import require_training_storage
    from multimodal_jev.context_policy_artifacts import read_execution_bundle,verify_native_adapter
    from multimodal_jev.context_runtime_profile import PROFILE
    from multimodal_jev.policy_identity import checkpoint_identity
    require_training_storage(out)
    for p,h in config['pins'].items():
        if file_hash(p)!=h:raise ValueError('runtime dependency changed')
    if (len(config['physical_devices'])!=1 or
            sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))!=
            ['davinci'+str(config['physical_devices'][0])]):
        raise ValueError('one isolated NPU required')
    if checkpoint_identity(config['model'])[0]!=config['base_id']:
        raise ValueError('base identity differs')
    bundle=read_execution_bundle(config['snapshot'],expected_policy_id=config['policy_id'],
        expected_contract=config['contract'],expected_base_id=config['base_id'])
    if (bundle['contract']['runtime_profile']!=PROFILE or
            bundle['contract']['request_version']!='jev-context-exclusive_event-v1' or
            bundle['contract']['token']['registration_status']!='admitted'):
        raise ValueError('admitted named task runtime required')
    verify_native_adapter(config['adapter'],bundle)
    rows,receipts,proof=load_inputs(config);indices=panel_indices(rows,receipts)
    encode=audited_encoder(config,proof)
    torch.set_num_threads(2);torch.npu.set_device(0)
    torch.manual_seed(317);torch.npu.manual_seed_all(317)
    device=torch.device('npu',0);records=[]
    if phase=='hf':
        from multimodal_jev.asr import load_runtime
        from multimodal_jev.sft import initialize_adapter
        from multimodal_jev.context_decision_head import ContextDecisionHead
        from multimodal_jev.context_decision_model import ContextDecisionModel
        from multimodal_jev.context_policy_snapshot import load_policy
        from multimodal_jev.context_runtime_profile import install_actor
        _,base=load_runtime(config['model'],device,'bfloat16')
        adapted=initialize_adapter(base,dict(lora_rank=4,lora_alpha=8,lora_execution_dtype='bfloat16')).eval()
        head=ContextDecisionHead(1024,8,1024,.1).to(device=device,dtype=torch.bfloat16).eval()
        model=ContextDecisionModel(adapted.get_base_model().model,head,151706).eval()
        load_policy(config['snapshot'],model,expected_policy_id=config['policy_id'],
            expected_contract=config['contract'],expected_base_id=config['base_id'])
        handle,runtime=install_actor(adapted,model,PROFILE)
        commit_json(out/'hf-runtime.json',runtime)
        for i in indices:
            row,receipt,value=encode((rows[i],receipts[i]))
            with torch.no_grad():
                probability=model({k:v.to(device) for k,v in value.inputs.items()},value.positions.to(device)).cpu()[0].tolist()
            identity=dict(policy_id=config['policy_id'],positions=receipt['positions'][0],
                inputs={k:v for k,v in receipt['tensors'].items() if k!='positions'})
            records.append(dict(id=row['id'],request_identity=identity,prob=probability))
    elif phase=='vllm':
        from multimodal_jev.context_resident_rollout import FixedContextRollout
        ec={k:config[k] for k in ('model','base_id','snapshot','policy_id','contract','adapter','physical_devices')}
        ec.update(tensor_parallel_size=1,max_model_len=4096,max_decisions=31)
        engine=FixedContextRollout(ec)
        try:
            commit_json(out/'vllm-runtime.json',engine.runtime)
            for i in indices:
                row,receipt,value=encode((rows[i],receipts[i]))
                records.append(checked_prediction(row,receipt,
                    engine.score(value.inputs,value.positions,audio_token_id=151676),config['policy_id']))
        finally:engine.close()
    else:raise ValueError('explicit backend required')
    report=dict(policy_id=config['policy_id'],indices=indices,records=records,
                input_manifest_sha256=config['input_manifest']['sha256'])
    commit_json(out/(phase+'.json'),report)
    if phase=='vllm':
        result=compare(json.loads((out/'hf.json').read_text()),report)
        commit_json(out/'comparison.json',result)
        if not result['exact']:raise ValueError('FSC backend probability parity failed')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--phase',required=True,choices=['hf','vllm'])
    a=p.parse_args();a.output.mkdir(exist_ok=True)
    run(json.loads(a.config.read_text()),a.output,a.phase)

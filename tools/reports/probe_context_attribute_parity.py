"""Finite label-free HF/vLLM probe for compact native attribute requests.

Run backends in separate processes on an exclusively leased NPU. This is a
three-audio/six-request diagnostic, not full task or training admission.
"""
import argparse
import json
import math
from pathlib import Path
from multimodal_jev.baseline import file_hash
from multimodal_jev.slurp_training import content_hash
from tools.reports.context_attribute_actor_candidate import REFERENCE_ID, CATALOG_SHA
from tools.reports.build_context_attribute_candidate import digest, request


def validate(config):
    required={'pins','model','base_id','snapshot','adapter','policy_id','contract',
              'physical_devices','catalog','audio','cache_root','source_root'}
    if set(config)!=required:raise ValueError('Exact label-free parity configuration required')
    if (config['policy_id']!=REFERENCE_ID or len(config['physical_devices'])!=1
            or type(config['physical_devices'][0]) is not int or not 0<=config['physical_devices'][0]<8):
        raise ValueError('Frozen Stage1 policy and one explicit device required')
    for path,h in config['pins'].items():
        if file_hash(path)!=h:raise ValueError('Pinned diagnostic dependency changed: '+path)
    menu=json.loads(Path(config['catalog']).read_text())
    if digest(menu)!=CATALOG_SHA:raise ValueError('Frozen attribute catalog required')
    assets=config['audio']
    if len(assets)!=3 or len({a['id'] for a in assets})!=3:
        raise ValueError('Three distinct predeclared TRAIN audio anchors required')
    for a in assets:
        if (set(a)!={'id','audio','sha256','samples','waveform_sha256'} or
                type(a['samples']) is not int or a['samples']<1 or file_hash(a['audio'])!=a['sha256']):
            raise ValueError('Complete pinned label-free audio anchor required')
    from multimodal_jev.context_policy_artifacts import read_execution_bundle,verify_native_adapter
    from multimodal_jev.cached_context_preparation import processor_cache_identity
    from multimodal_jev.context_runtime_profile import PROFILE
    bundle=read_execution_bundle(config['snapshot'],expected_policy_id=REFERENCE_ID,
        expected_contract=config['contract'],expected_base_id=config['base_id'])
    verify_native_adapter(config['adapter'],bundle)
    if (bundle['contract']['runtime_profile']!=PROFILE or bundle['contract']['token']['id']!=151706
            or bundle['contract']['request_version']!='jev-context-exclusive_event-v1'):
        raise ValueError('Original admitted Stage1 artifact contract must be preserved')
    return menu,bundle,processor_cache_identity(config['snapshot'])


def checked(record, *, policy_id, decisions):
    identity=record['request_identity'];p=record['prob']
    if (record['policy_id']!=policy_id or identity['policy_id']!=policy_id
            or record['request_id']!=content_hash(identity) or len(p)!=decisions
            or len(identity['positions'])!=decisions
            or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in p)):
        raise ValueError('Bound complete finite probability vector required')


def compare(hf, vl):
    if (hf['policy_id']!=REFERENCE_ID or vl['policy_id']!=REFERENCE_ID
            or hf['input_panel']!=vl['input_panel'] or len(hf['input_panel'])!=6
            or len(hf['records'])!=6 or len(vl['records'])!=6
            or hf['backend']!='hf' or vl['backend']!='vllm'):
        raise ValueError('Same frozen policy and complete six-request panel required')
    panel=hf['input_panel']
    if (len({(r['id'],r['arm']) for r in panel})!=6
            or {r['arm'] for r in panel}!={'direct','q_input'}
            or len({r['id'] for r in panel})!=3
            or any({r['arm'] for r in panel if r['id']==uid}!={'direct','q_input'} for uid in {r['id'] for r in panel})):
        raise ValueError('Both arms for each unique audio required')
    maximum=0.;flips=0;saturated=0;by_arm={}
    for expected,a,b in zip(panel,hf['records'],vl['records'],strict=True):
        n=60 if expected['arm']=='direct' else 124
        for record in (a,b):
            checked(record,policy_id=REFERENCE_ID,decisions=n)
            if any(record[k]!=expected[k] for k in ('id','arm','request_sha256','decision_ids')):
                raise ValueError('Parity record differs from declared input panel')
            if len(record['decision_ids'])!=n or len(set(record['decision_ids']))!=n:
                raise ValueError('Complete ordered decision IDs required')
        if a['request_identity']!=b['request_identity'] or a['request_id']!=b['request_id']:
            raise ValueError('HF/vLLM consumed different actual tensors')
        delta=max(abs(x-y) for x,y in zip(a['prob'],b['prob'],strict=True))
        maximum=max(maximum,delta);by_arm[expected['arm']]=max(by_arm.get(expected['arm'],0.),delta)
        flips+=sum((x>=.5)!=(y>=.5) for x,y in zip(a['prob'],b['prob'],strict=True))
        saturated+=sum(p in (0.,1.) for p in a['prob'])
    return dict(requests=6,probability_max_abs=maximum,by_arm_max_abs=by_arm,
        threshold_flips=flips,saturated_probabilities=saturated,exact=maximum==0 and flips==0,
        finite_log_policy_supported=saturated==0,
        scope='Three TRAIN anchors / two input arms; no gradients, optimizer, full population or efficacy admission')


def prepare(config, output):
    import hashlib
    import numpy as np
    import torch
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.encoded_audio_cache import EncodedAudioCache
    from tools.reports.context_attribute_input_candidate import AttributeInputs
    menu,bundle,processor_id=validate(config)
    rows=[];panel=[]
    for arm in ('direct','q_input'):
        adapter=AttributeInputs(menu,arm,source_root=config['source_root'],
            processor_factory=lambda:Qwen3ASRProcessor.from_pretrained(bundle['processor'],local_files_only=True),
            cache=EncodedAudioCache(config['cache_root']),processor_sha256=processor_id,
            decision_token_id=151706,audio_token_id=151676,max_tokens=4096)
        for asset in config['audio']:
            wave=load_audio(asset['audio'])
            if len(wave)!=asset['samples'] or hashlib.sha256(np.ascontiguousarray(wave,dtype=np.float32).tobytes()).hexdigest()!=asset['waveform_sha256']:
                raise ValueError('Complete original anchor waveform required')
            req=request(asset['audio'],menu,arm=arm)
            value=adapter((req,asset['sha256']))
            entry=dict(id=asset['id'],arm=arm,request_sha256=digest(req.model_input()),
                decision_ids=adapter.contract['decision_ids'])
            panel.append(entry);rows.append((entry,value))
    return rows,panel


def run(config,out,phase):
    if phase not in ('prepare','hf','vllm'):raise ValueError('Known backend phase required')
    import torch
    from multimodal_jev.local_runtime_storage import require_training_storage
    from multimodal_jev.local_checkpoint import commit_json
    from multimodal_jev.span_request_identity import tensor_identity
    require_training_storage(out);require_training_storage(config['cache_root'])
    rows,panel=prepare(config,out)
    if phase=='prepare':
        if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU preparation must not receive NPU devices')
        result=dict(policy_id=REFERENCE_ID,input_panel=panel,inputs=[dict(entry,
            positions=value.positions.tolist(),tensors={k:tensor_identity(v) for k,v in value.inputs.items()}) for entry,value in rows])
        commit_json(out/'prepared.json',result);return result
    import torch_npu
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.context_runtime_profile import PROFILE,install_actor
    if sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))!=['davinci'+str(config['physical_devices'][0])]:
        raise ValueError('One exclusively leased NPU required')
    if checkpoint_identity(config['model'])[0]!=config['base_id']:raise ValueError('Base model differs')
    torch.set_num_threads(2);torch.npu.set_device(0);torch.manual_seed(317);torch.npu.manual_seed_all(317)
    records=[]
    if phase=='hf':
        from multimodal_jev.asr import load_runtime
        from multimodal_jev.sft import initialize_adapter
        from multimodal_jev.context_decision_head import ContextDecisionHead
        from multimodal_jev.context_decision_model import ContextDecisionModel
        from multimodal_jev.context_policy_snapshot import load_policy
        device=torch.device('npu',0)
        _,base=load_runtime(config['model'],device,'bfloat16')
        adapted=initialize_adapter(base,dict(lora_rank=4,lora_alpha=8,lora_execution_dtype='bfloat16')).eval()
        head=ContextDecisionHead(1024,8,1024,.1).to(device=device,dtype=torch.bfloat16).eval()
        model=ContextDecisionModel(adapted.get_base_model().model,head,151706).eval()
        load_policy(config['snapshot'],model,expected_policy_id=REFERENCE_ID,
            expected_contract=config['contract'],expected_base_id=config['base_id'])
        handle,runtime=install_actor(adapted,model,PROFILE);commit_json(out/'hf-runtime.json',runtime)
        for entry,value in rows:
            with torch.no_grad():prob=model({k:v.to(device) for k,v in value.inputs.items()},value.positions.to(device)).cpu()[0].tolist()
            identity=dict(policy_id=REFERENCE_ID,positions=value.positions[0].tolist(),
                inputs={k:tensor_identity(v) for k,v in sorted(value.inputs.items())})
            records.append(dict(entry,policy_id=REFERENCE_ID,request_identity=identity,request_id=content_hash(identity),prob=prob))
    else:
        from multimodal_jev.context_resident_rollout import FixedContextRollout
        ec={k:config[k] for k in ('model','base_id','snapshot','policy_id','contract','adapter','physical_devices')}
        ec.update(tensor_parallel_size=1,max_model_len=4096,max_decisions=124)
        engine=FixedContextRollout(ec)
        try:
            commit_json(out/'vllm-runtime.json',engine.runtime)
            for entry,value in rows:records.append(dict(engine.score(value.inputs,value.positions,audio_token_id=151676),**entry))
        finally:engine.close()
    result=dict(backend=phase,policy_id=REFERENCE_ID,input_panel=panel,records=records)
    commit_json(out/(phase+'.json'),result)
    if phase=='vllm':
        parity=compare(json.loads((out/'hf.json').read_text()),result);commit_json(out/'parity.json',parity)
        if not parity['exact'] or not parity['finite_log_policy_supported']:raise ValueError('Exact finite probability parity required')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--phase',required=True,choices=['prepare','hf','vllm']);a=p.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=True)
    run(json.loads(Path(a.config).read_text()),out,a.phase)

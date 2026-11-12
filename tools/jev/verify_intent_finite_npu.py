#!/usr/bin/env python3
"""Audit first finite-support intent update against independent HF/vLLM export.

This checks actual shard tensors and probabilities, not just passed receipts.
It is not proof of long-run RL efficacy or Ray orchestration.
"""
import argparse
import json
from pathlib import Path
import torch
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.span_parity import compare_span_policies
from multimodal_jev.span_native_actor import ACTOR_SHA256
from multimodal_jev.slurp_intent_pilot import load_pilot
from multimodal_jev.intent_actor_inputs import schedule
from multimodal_jev.intent_gate_identity import verify_population


def verify(root,suite):
    root,suite=map(Path,(root,suite));read=lambda p:json.loads(p.read_text());config=read(root/'config.json');launch=read(root/'launch.json')
    for path,h in config['pins'].items():
        if file_hash(path)!=h:raise ValueError('training input changed')
    if file_hash(root/'config.json')!=launch['config_sha256'] or file_hash(root/'source-manifest.json')!=launch['source_manifest_sha256']:
        raise ValueError('training source/config changed')
    for path,h in read(root/'source-manifest.json').items():
        if file_hash(root/'source'/path)!=h:raise ValueError('training snapshot changed')
    from train_intent_finite_npu import validate_config
    validate_config(config)
    items,admission=load_pilot(config['pilot'],config['manifest_sha256'],config['arm'])
    selected=schedule(items,config['seed'],config['step'],config['global_requests'])
    ids=[r['id'] for r in selected]
    selection=read(root/'run/selection.json')
    if selection['recording_ids']!=ids or selection['source_groups']!=[r['target']['source_group'] for r in selected]:
        raise ValueError('training schedule differs')
    adapter_id,adapter_files=checkpoint_identity(root/'run/adapter')
    base_id=checkpoint_identity(config['model_path'])[0];initial_id=base_id+':'+checkpoint_identity(config['initial_adapter'])[0]
    suite_launch=read(suite/'launch.json');suite_config=read(suite/'config.json')
    if file_hash(suite/'config.json')!=suite_launch['config_sha256'] or file_hash(suite/'source-manifest.json')!=suite_launch['source_manifest_sha256']:
        raise ValueError('suite config/source changed')
    for path,h in suite_config['pins'].items():
        if file_hash(path)!=h:raise ValueError('suite input changed')
    for path,h in read(suite/'source-manifest.json').items():
        if file_hash(suite/'source'/path)!=h:raise ValueError('suite snapshot changed')
    if (suite_config['adapter']!=str(root/'run/adapter') or suite_config['model']!=config['model_path']
            or suite_config['runtime_profile']!=config['runtime_profile']):raise ValueError('post-update gate differs')
    for name in ('terminal.json','hf/terminal.json','vllm/terminal.json'):
        if read(suite/'run'/name)['exit_code']!=0:raise ValueError('independent phase failed')
    ready=read(suite/'run/READY.json');hf=read(suite/'run/hf/result.json');vl=read(suite/'run/vllm/result.json')
    if not ready['passed'] or ready['result_sha256']!=file_hash(suite/'run/vllm/result.json'):raise ValueError('suite not ready')
    for result in (hf,vl):
        if not result['passed'] or result['base_id']!=base_id or result['adapter_id']!=adapter_id:raise ValueError('independent policy differs')
    inputs=Path(suite_config['inputs'])
    if suite_config['pins'].get(str(inputs))!=file_hash(inputs):raise ValueError('pinned processor inputs required')
    artifact=read(inputs);expected={}
    for arm in ('D','Q'):
        population,_=load_pilot(config['pilot'],config['manifest_sha256'],arm)
        expected.update({item['id']+'::'+arm:item['request'].model_input() for item in population})
    for result in (hf,vl):verify_population(artifact,expected,result['records'],base_id+':'+adapter_id)
    if (ready['requests']!=len(expected) or ready['base_id']!=base_id
            or ready['adapter_id']!=adapter_id or suite_config['base_id']!=base_id
            or suite_config['adapter_id']!=adapter_id):raise ValueError('readiness policy/population differs')
    refs={r['id']:r for r in hf['records']};observed={r['id']:r for r in vl['records']}
    expected_population={r['id']+'::'+a for r in items for a in ('D','Q')}
    if (set(refs)!=expected_population or set(observed)!=expected_population
            or len(hf['records'])!=len(expected_population) or len(vl['records'])!=len(expected_population)):
        raise ValueError('all learning D/Q input probabilities required')
    audits=[]
    for uid,ref in refs.items():
        actual=observed[uid]
        if ref['request_id']!=actual['request_id'] or ref['request_identity']!=actual['request_identity']:raise ValueError('independent request mismatch')
        audit=compare_span_policies(ref['candidate_logprobs'],actual['candidate_logprobs'])
        if not audit['passed']:raise ValueError('export HF/vLLM parity failed')
        audits.append(audit)
    files={};all_rank_ids=[];wire_arm='D' if config['arm']=='D' else 'Q'
    global_weight=sum(r['target']['request_weight'] for r in selected)
    rank_weights=[sum(r['target']['request_weight'] for r in selected[i:i+2]) for i in range(0,8,2)]
    for rank in range(4):
        out=root/'run'/f'rank-{rank}';result=read(out/'result.json');local_ids=ids[rank*2:rank*2+2]
        if (read(out/'terminal.json')['exit_code']!=0 or result['passed'] is not True or result['rank']!=rank
                or result['world_size']!=4 or result['updates']!=1 or result['arm']!=config['arm']
                or result['actor_sha256']!=ACTOR_SHA256 or result['base_id']!=base_id
                or result['initial_policy_id']!=initial_id or result['adapter_id']!=adapter_id
                or result['recording_ids']!=local_ids or not result['frozen_base_unchanged'] or not result['cross_rank_export_exact']):
            raise ValueError('rank contract differs')
        expected_admission=dict(passed=True,world_size=4,global_requests=8,global_effective_weight=global_weight,rank_effective_weights=rank_weights)
        if result['admission']!=expected_admission:raise ValueError('global weighting differs')
        if len(result['execution_dtypes'])!=112 or any(v!=['torch.bfloat16','torch.bfloat16'] for v in result['execution_dtypes'].values()):
            raise ValueError('112 BF16 adapter executions required')
        saved=torch.load(out/'optimizer-local.pt',map_location='cpu',weights_only=True)
        if (saved['rank']!=rank or saved['world_size']!=4 or saved['updates']!=1
                or len(saved['names'])!=112 or set(saved['names'])!=set(saved['parameters'])):raise ValueError('optimizer layout differs')
        if any(p.dtype!=torch.float32 or not torch.isfinite(p).all() for p in saved['parameters'].values()):raise ValueError('finite FP32 masters required')
        adam=saved['optimizer']['state']
        if len(adam)!=112:raise ValueError('Adam coverage differs')
        for s in adam.values():
            if s['step'].item()!=1:raise ValueError('Adam cursor differs')
            for name in ('exp_avg','exp_avg_sq'):
                if s[name].dtype!=torch.float32 or not torch.isfinite(s[name]).all():raise ValueError('finite FP32 moments required')
            if (s['exp_avg_sq']<0).any():raise ValueError('negative Adam second moment')
        after=read(out/'after.json')
        if [r['id'] for r in after]!=local_ids:raise ValueError('after-update population differs')
        for row,item in zip(after,selected[rank*2:rank*2+2],strict=True):
            ref=refs[row['id']+'::'+wire_arm]
            expected_request=json.loads(json.dumps(item['request'].model_input()))
            if ref['request_identity']['request']!=expected_request:raise ValueError('export reloaded different request')
            if row['candidate_logprobs']!=ref['candidate_logprobs']:raise ValueError('FSDP/HF export reload not exact')
        all_rank_ids.extend(local_ids)
        for name in ('result.json','terminal.json','selection.json','before.json','after.json','optimizer-local.pt'):files[str(out/name)]=file_hash(out/name)
    if len(set(all_rank_ids))!=8:raise ValueError('unique global sources required')
    for name in ('READY.json','terminal.json','hf/result.json','vllm/result.json'):files[str(suite/'run'/name)]=file_hash(suite/'run'/name)
    return dict(passed=True,root=str(root),suite=str(suite),adapter_id=adapter_id,adapter_files=adapter_files,
        updates=1,arm=config['arm'],recordings=8,full_gate_requests=len(expected_population),
        exact_export_reload=True,maximum_logprob_error=max(a['maximum_logprob_error'] for a in audits),
        files=files,scope='First finite-support intent update: native verl/FSDP state and independent BF16 export/HF/vLLM verified. No held-out task gain, continuous trainer or transfer result.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--suite',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();atomic_json(a.output,verify(a.root,a.suite))

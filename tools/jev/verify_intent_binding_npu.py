#!/usr/bin/env python3
"""Audit a binding-augmented update against all712 independent HF/vLLM inputs.

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
from multimodal_jev.intent_binding_resume import expected_selection
from multimodal_jev.intent_binding_inputs import load
from multimodal_jev.intent_binding_gate import verify_records,authenticate_artifacts
from multimodal_jev.intent_batch_layout import rank_slice


def verify_supervision(selected,supervision,behaviors):
    if len(supervision)!=len(selected):raise ValueError('complete external view supervision required')
    for item,row in zip(selected,supervision,strict=True):
        target=item['target']
        expected=dict(request_id=behaviors[item['gate_id']]['request_id'],
            targets=[target['targets']],supervised=[target['supervised']],
            span_weights=torch.tensor([target['span_weights']],dtype=torch.float32).tolist(),
            request_weight=target['request_weight'])
        if row!=expected:raise ValueError('recorded supervision differs from remapped gold')


def verify_before(selected,before,behaviors):
    if len(before['old'])!=len(selected) or len(before['actor'])!=len(selected):
        raise ValueError('complete before-update rows required')
    for item,old,actual in zip(selected,before['old'],before['actor'],strict=True):
        ref=behaviors[item['gate_id']]['candidate_logprobs'];flat=[p for span in ref for p in span]
        # The stored behavior is FP32, matching the controller tensor cast.
        expected=torch.tensor(flat,dtype=torch.float32).tolist()
        if len(old)!=len(flat) or old!=expected or len(actual)!=len(flat):
            raise ValueError('before-update behavior differs')
        offset=0;split=[]
        for span in ref:
            split.append(actual[offset:offset+len(span)]);offset+=len(span)
        if not compare_span_policies(split,ref)['passed']:raise ValueError('before-update actor differs')


def verify(root,suite):
    root,suite=map(Path,(root,suite));read=lambda p:json.loads(p.read_text());config=read(root/'config.json');launch=read(root/'launch.json')
    for path,h in config['pins'].items():
        if file_hash(path)!=h:raise ValueError('training input changed')
    if file_hash(root/'config.json')!=launch['config_sha256'] or file_hash(root/'source-manifest.json')!=launch['source_manifest_sha256']:
        raise ValueError('training source/config changed')
    for path,h in read(root/'source-manifest.json').items():
        if file_hash(root/'source'/path)!=h:raise ValueError('training snapshot changed')
    from multimodal_jev.intent_binding_resume import validate_config
    validate_config(config)
    if launch['image']!=config['image'] or launch['physical_devices']!=[4,5,6,7]:
        raise ValueError('fixed DP4 training launch required')
    updates=config['step']+1
    if config['step']>0:
        from multimodal_jev.intent_binding_resume import authenticate
        lineage=authenticate(config)
        if read(root/'run/resume-lineage.json')!=lineage:raise ValueError('recorded continuation lineage differs')
    selected,expected_selection_receipt=expected_selection(config)
    ids=[r['id'] for r in selected]
    selection=read(root/'run/selection.json')
    if any(selection.get(k)!=v for k,v in expected_selection_receipt.items()):
        raise ValueError('binding source/view schedule differs')
    adapter_id,adapter_files=checkpoint_identity(root/'run/adapter')
    base_id=checkpoint_identity(config['model_path'])[0];initial_id=base_id+':'+checkpoint_identity(config['initial_adapter'])[0]
    suite_launch=read(suite/'launch.json');suite_config=read(suite/'config.json')
    if suite_launch['image']!=config['image'] or suite_launch['physical_devices']!=[4]:
        raise ValueError('fixed independent TP1 launch required')
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
    if suite_config.get('training_root')!=str(root) or suite_config['population']!=dict(recordings=89,requests=712):
        raise ValueError('full updated binding suite required')
    artifact=read(inputs)
    populations={a:load(config['binding_release'],config['binding_manifest_sha256'],a)[0] for a in ('D','Q')}
    indices,gate_audit=verify_records(populations,artifact,hf,vl,base_id+':'+adapter_id)
    if (ready['requests']!=712 or ready['base_id']!=base_id or ready['adapter_id']!=adapter_id
            or suite_config['base_id']!=base_id or suite_config['adapter_id']!=adapter_id):
        raise ValueError('readiness policy/population differs')
    refs=indices[0]
    _,before_indices,before_gate=authenticate_artifacts(config,initial_id)
    if selection['gate']!=before_gate:raise ValueError('training behavior gate differs')
    supervision=read(root/'run/supervision.json')
    verify_supervision(selected,supervision,before_indices[1])
    files={str(root/'run'/name):file_hash(root/'run'/name) for name in ('selection.json','supervision.json')}
    all_rank_ids=[]
    global_weight=sum(r['target']['request_weight'] for r in selected)
    rank_weights=[sum(r['target']['request_weight'] for r in selected[rank_slice(config,i)]) for i in range(4)]
    for rank in range(4):
        out=root/'run'/f'rank-{rank}';result=read(out/'result.json');local_ids=ids[rank_slice(config,rank)]
        if (read(out/'terminal.json')['exit_code']!=0 or result['passed'] is not True or result['rank']!=rank
                or result['world_size']!=4 or result['updates']!=updates or result['arm']!=config['arm']
                or result['actor_sha256']!=ACTOR_SHA256 or result['base_id']!=base_id
                or result['initial_policy_id']!=initial_id or result['adapter_id']!=adapter_id
                or result['view_ids']!=local_ids
                or result['source_ids']!=selection['source_ids'][rank*2:(rank+1)*2]
                or result['gate_ids']!=selection['gate_ids'][rank_slice(config,rank)] or not result['frozen_base_unchanged'] or not result['cross_rank_export_exact']):
            raise ValueError('rank contract differs')
        rank_selection=read(out/'selection.json')
        if rank_selection!=dict(selection,rank=rank,local_view_ids=local_ids):raise ValueError('rank selection differs')
        before=read(out/'before.json')
        verify_before(selected[rank_slice(config,rank)],before,before_indices[1])
        expected_admission=dict(passed=True,world_size=4,global_requests=config['global_requests'],global_effective_weight=global_weight,rank_effective_weights=rank_weights)
        if result['admission']!=expected_admission or before['admission']!=expected_admission:
            raise ValueError('global weighting differs')
        if len(result['execution_dtypes'])!=112 or any(v!=['torch.bfloat16','torch.bfloat16'] for v in result['execution_dtypes'].values()):
            raise ValueError('112 BF16 adapter executions required')
        saved=torch.load(out/'optimizer-local.pt',map_location='cpu',weights_only=True)
        from multimodal_jev.intent_optimizer import validate_saved_optimizer
        validate_saved_optimizer(config,saved['optimizer'])
        if (saved['rank']!=rank or saved['world_size']!=4 or saved['updates']!=updates
                or len(saved['names'])!=112 or set(saved['names'])!=set(saved['parameters'])):raise ValueError('optimizer layout differs')
        if any(p.dtype!=torch.float32 or not torch.isfinite(p).all() for p in saved['parameters'].values()):raise ValueError('finite FP32 masters required')
        adam=saved['optimizer']['state']
        if len(adam)!=112:raise ValueError('Adam coverage differs')
        for s in adam.values():
            if s['step'].item()!=updates:raise ValueError('Adam cursor differs')
            for name in ('exp_avg','exp_avg_sq'):
                if s[name].dtype!=torch.float32 or not torch.isfinite(s[name]).all():raise ValueError('finite FP32 moments required')
            if (s['exp_avg_sq']<0).any():raise ValueError('negative Adam second moment')
        if config['step']>0:
            prior_path=Path(config['previous_root'])/'run'/f'rank-{rank}'/'optimizer-local.pt'
            prior=torch.load(prior_path,map_location='cpu',weights_only=True)
            restored=torch.load(out/'restored-local.pt',map_location='cpu',weights_only=True)
            if restored['names']!=prior['names'] or set(restored['parameters'])!=set(prior['parameters']):
                raise ValueError('restored master layout differs')
            for name,value in prior['parameters'].items():
                if not torch.equal(restored['parameters'][name],value):raise ValueError('restored master not exact')
            a,b=restored['optimizer'],prior['optimizer']
            if a['param_groups']!=b['param_groups'] or set(a['state'])!=set(b['state']):raise ValueError('restored Adam layout differs')
            for key,fields in b['state'].items():
                if set(a['state'][key])!=set(fields):raise ValueError('restored Adam fields differ')
                for field,value in fields.items():
                    if not torch.equal(a['state'][key][field],value):raise ValueError('restored Adam not exact')
            receipt=read(out/'optimizer-resume.json')
            if (result['restored']!=receipt or receipt['previous_updates']!=config['step']
                    or not receipt['exact_masters'] or not receipt['exact_adam']):raise ValueError('restore receipt differs')
            for name in ('restored-local.pt','optimizer-resume.json'):files[str(out/name)]=file_hash(out/name)
        after=read(out/'after.json')
        if [r['id'] for r in after]!=local_ids:raise ValueError('after-update population differs')
        for row,item in zip(after,selected[rank_slice(config,rank)],strict=True):
            ref=refs[item['gate_id']]
            if row['previous_request_id']!=before_indices[1][item['gate_id']]['request_id']:
                raise ValueError('updated row behavior identity differs')
            expected_request=json.loads(json.dumps(item['request'].model_input()))
            if ref['request_identity']['request']!=expected_request:raise ValueError('export reloaded different request')
            if row['candidate_logprobs']!=ref['candidate_logprobs']:raise ValueError('FSDP/HF export reload not exact')
        all_rank_ids.extend(local_ids)
        for name in ('result.json','terminal.json','selection.json','before.json','after.json','optimizer-local.pt'):files[str(out/name)]=file_hash(out/name)
    if len(all_rank_ids)!=config['global_requests'] or len(set(all_rank_ids))!=config['global_requests']:raise ValueError('unique global views required')
    for name in ('READY.json','terminal.json','hf/result.json','vllm/result.json'):files[str(suite/'run'/name)]=file_hash(suite/'run'/name)
    return dict(schema='intent-binding-update-verified-v1',passed=True,root=str(root),suite=str(suite),adapter_id=adapter_id,adapter_files=adapter_files,
        updates=updates,arm=config['arm'],source_uses=8,view_uses=32,full_gate_requests=712,
        binding_manifest_sha256=config['binding_manifest_sha256'],
        exact_export_reload=True,maximum_logprob_error=gate_audit['maximum_logprob_error'],
        files=files,scope='Binding-augmented intent update: native verl/FSDP state, applicable exact restore and independent BF16 export/HF/vLLM verified. No held-out task gain, continuous Ray trainer or transfer result.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--suite',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();atomic_json(a.output,verify(a.root,a.suite))

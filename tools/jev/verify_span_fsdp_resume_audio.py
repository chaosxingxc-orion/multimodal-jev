#!/usr/bin/env python3
"""Audit the fixed-topology second update and its independent full suite."""
import argparse
import json
from pathlib import Path
import torch
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.span_parity import compare_span_policies


def verify_resume(root,suite):
    root,suite=Path(root),Path(suite)
    read=lambda p:json.loads(p.read_text())
    launch=read(root/'launch.json');config=read(root/'config.json')
    from verify_span_fsdp_audio import verify
    previous=verify(config['previous_fsdp_root'],config['previous_fsdp_suite'])
    if Path(config['initial_adapter']).resolve()!=(Path(config['previous_fsdp_root'])/'run/adapter').resolve():
        raise ValueError('parent adapter mismatch')
    if file_hash(root/'config.json')!=launch['config_sha256'] or file_hash(root/'source-manifest.json')!=launch['source_manifest_sha256']:
        raise ValueError('run configuration/source manifest changed')
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('training asset changed')
    source=read(root/'source-manifest.json')
    for path,digest in source.items():
        if file_hash(root/'source'/path)!=digest:raise ValueError('source changed')
    if read(suite/'run/terminal.json')['exit_code']!=0 or read(suite/'run/READY.json')['status']!='passed':
        raise ValueError('independent full suite did not pass')
    hf=read(suite/'run/hf/result.json');vllm=read(suite/'run/vllm/result.json')
    adapter_id,files=checkpoint_identity(root/'run/adapter')
    if hf['adapter_id']!=adapter_id or vllm['adapter_id']!=adapter_id or not vllm['passed']:
        raise ValueError('independent reference/version mismatch')
    if len(hf['records'])!=48 or len(vllm['records'])!=48:raise ValueError('48 recordings required')
    references={r['id']:r for r in hf['records']}
    metrics=[]
    for r in vllm['records']:
        independent=references[r['id']]
        if (r['request_id']!=independent['request_id'] or r['request_identity']!=independent['request_identity']
                or r['hf_candidate_logprobs']!=[p[0] for p in independent['candidate_logprobs']]):
            raise ValueError('vLLM gate not bound to independent HF')
        audit=compare_span_policies(r['hf_candidate_logprobs'],r['candidate_logprobs'])
        if not audit['passed']:raise ValueError('vLLM parity failed')
        metrics.append(audit)
    results=[];recordings=[];errors=[];pins={}
    for rank in range(4):
        out=root/'run'/f'rank-{rank}'
        r=read(out/'result.json')
        if (read(out/'terminal.json')['exit_code']!=0 or r['status']!='passed'
                or r['rank']!=rank or r['world_size']!=4 or r['optimizer_updates']!=2
                or r['adapter_id']!=adapter_id or not r['frozen_base_unchanged']
                or not r['cross_rank_export_exact']):raise ValueError('rank update contract failed')
        if len(r['execution_dtypes'])!=112 or any(v!=['torch.bfloat16','torch.bfloat16'] for v in r['execution_dtypes'].values()):
            raise ValueError('112 BF16 A/B computations required')
        if not all(a['passed'] for a in r['admission']):raise ValueError('before-update admission failed')
        selection=read(out/'selection.json')
        if selection['batch_index']!=config['batch_index']+rank or selection['recording_ids']!=r['recordings']:
            raise ValueError('balanced rank selection differs')
        restored=read(out/'optimizer-resume.json')
        if (restored.get('previous_updates')!=1 or restored.get('rank')!=rank
                or restored.get('world_size')!=4 or restored.get('master_shards')!=112
                or restored.get('exact_masters') is not True or restored.get('exact_adam') is not True):
            raise ValueError('native optimizer restore receipt missing')
        from multimodal_jev.span_fsdp_resume import validate_cursor
        prior=Path(config['previous_fsdp_root'])/'run'/f'rank-{rank}'
        validate_cursor(read(prior/'selection.json'),selection,4)
        original=torch.load(prior/'optimizer-local.pt',map_location='cpu',weights_only=True)
        actual=torch.load(out/'restored-local.pt',map_location='cpu',weights_only=True)
        if (actual['names']!=original['names'] or actual['parameters'].keys()!=original['parameters'].keys()
                or actual['optimizer']['param_groups']!=original['optimizer']['param_groups']
                or actual['optimizer']['state'].keys()!=original['optimizer']['state'].keys()):
            raise ValueError('actual restored shard layout differs')
        for name,value in original['parameters'].items():
            if actual['parameters'][name].dtype!=value.dtype or not torch.equal(actual['parameters'][name],value):
                raise ValueError('actual restored FP32 master differs')
        for key,fields in original['optimizer']['state'].items():
            if actual['optimizer']['state'][key].keys()!=fields.keys():
                raise ValueError('actual restored Adam fields differ')
            for field,value in fields.items():
                observed=actual['optimizer']['state'][key][field]
                if observed.dtype!=value.dtype or not torch.equal(observed,value):
                    raise ValueError('actual restored Adam tensor differs')
        if r['initial_policy_id']!=r['base_id']+':'+previous['adapter_id']:
            raise ValueError('restored execution policy differs from parent')
        state=torch.load(out/'optimizer-local.pt',map_location='cpu',weights_only=True)
        if state['world_size']!=4 or state['rank']!=rank or state['updates']!=2:
            raise ValueError('optimizer shard topology differs')
        if len(state['parameters'])!=112 or any(p.dtype!=torch.float32 for p in state['parameters'].values()):
            raise ValueError('FP32 master shards required')
        adam=state['optimizer']['state']
        if len(adam)!=112 or any(s['step'].item()!=2 or s['exp_avg'].dtype!=torch.float32 or s['exp_avg_sq'].dtype!=torch.float32 for s in adam.values()):
            raise ValueError('FP32 Adam shards at step two required')
        for post in read(out/'after.json'):
            ref=references[post['recording_id']]
            if post['input_request']!=ref['request_identity']['request']:
                raise ValueError('export reference input changed')
            error=max(abs(a-b) for x,y in zip(post['candidate_logprobs'],ref['candidate_logprobs'],strict=True)
                      for a,b in zip(x,y[0],strict=True))
            errors.append(error)
            if error!=0.:raise ValueError('FSDP post-update/HF export reload differs: '+str(error))
        results.append(r);recordings.extend(r['recordings'])
        pins.update({str(p):file_hash(p) for p in [out/'result.json',out/'terminal.json',out/'after.json',out/'optimizer-local.pt',out/'optimizer-resume.json',out/'restored-local.pt']})
    if len(set(recordings))!=8:raise ValueError('eight distinct training recordings required')
    for p in [suite/'run/READY.json',suite/'run/terminal.json',suite/'run/hf/result.json',suite/'run/vllm/result.json']:
        pins[str(p)]=file_hash(p)
    return dict(root=str(root),suite=str(suite),passed=True,physical_devices=[4,5,6,7],
        previous_adapter_id=previous['adapter_id'],optimizer_updates=2,
        adapter_id=adapter_id,adapter_files=files,training_recordings=8,acceptance_recordings=48,
        exact_fsdp_export_reload_maximum_error=max(errors),
        full_suite_maximum_logprob_error=max(a['maximum_logprob_error'] for a in metrics),
        full_suite_action_ratio_range=[min(a['joint_ratio_min'] for a in metrics),max(a['joint_ratio_max'] for a in metrics)],
        ranks=results,files=pins,scope='One resumed real-audio four-rank FSDP update, native Adam step two; exact independent HF export reload on eight inputs; '
        'full 48-input TP1 vLLM parity. Not uninterrupted-vs-resumed equivalence, TP4 rollout, complete Ray trainer, Omni30B or task-gain evidence.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--suite',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=verify_resume(args.root,args.suite);atomic_json(args.output,result)
    print(json.dumps({k:result[k] for k in ['passed','training_recordings','exact_fsdp_export_reload_maximum_error','full_suite_maximum_logprob_error','full_suite_action_ratio_range']}))

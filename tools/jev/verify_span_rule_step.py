#!/usr/bin/env python3
"""Audit a matched rule update and, optionally, independent HF reload.

This checks actual optimizer/checkpoint state and recorded local objectives.
Full cross-backend acceptance is separate and is never inferred here.
"""
import argparse
import json
from pathlib import Path
import torch
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.span_training_chain import checkpoint_receipt
from multimodal_jev.span_rule_objectives import expected_rule_reward, rule_sample_rewards
from multimodal_jev.span_policy import joint_logprobs
from multimodal_jev.span_parity import compare_span_policies
from multimodal_jev.policy_identity import checkpoint_identity
from infer_span_rule_baseline import select_rows


def verify(root, branch, hf_root=None):
    read=lambda p:json.loads(p.read_text())
    config=read(root/(branch+'.json'));run=root/branch
    launch=read(root/'launch.json');protocol=read(root/'protocol.json')
    if (read(run/'terminal.json')['exit_code']!=0
            or launch['config_sha256'][branch]!=file_hash(root/(branch+'.json'))
            or launch['protocol_sha256']!=file_hash(root/'protocol.json')
            or launch['source_manifest_sha256']!=file_hash(root/'source-manifest.json')
            or launch['controller_sha256']!=file_hash(root/'controller.py')):
        raise ValueError('run did not finish with pinned configuration')
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('pinned asset changed')
    if (config['request_ids']!=protocol['request_ids'] or config['seed']!=protocol['seed']
            or config['lr']!=protocol['lr'] or config['optimizer_start']!=protocol['optimizer_start']
            or config['num_updates']!=protocol['num_updates']
            or config['runtime_admission']!=protocol['runtime_admission']
            or config['arm']!=protocol['arm']
            or [config['method'],config['auxiliary_weight']] not in protocol['objectives']):
        raise ValueError('branch differs from common matched protocol')
    report=read(run/'update.json');before=read(run/'before.json');after=read(run/'after.json')
    receipt,_=checkpoint_receipt(run/'policy-1')
    if (receipt['completed_updates']!=1 or report['optimizer_updates']!=1
            or report['optimizer_start']!='fresh'
            or report['exported_policy_id']!=receipt['base_id']+':'+receipt['adapter_id']
            or after['policy_id']!=report['exported_policy_id']):
        raise ValueError('optimizer lineage differs')
    state=torch.load(run/'policy-1/optimizer.pt',map_location='cpu',weights_only=True)
    adam=state['optimizer']['optimizer']['state']
    if (len(adam)!=112 or any(int(s['step'])!=1 for s in adam.values())
            or any(v.dtype!=torch.float32 for s in adam.values() for v in s.values() if torch.is_tensor(v))
            or state['training_state']!=receipt['training_state']
            or state['completed_updates']!=1 or state['adapter_id']!=receipt['adapter_id']):
        raise ValueError('actual Adam state differs')
    rows={r['id']:r for r in select_rows(Path(config['release']))}
    selected=[rows[k] for k in config['request_ids']]
    batch=json.loads((Path(config['release'])/'schedule.json').read_text())['batches'][0]
    if config['request_ids']!=batch['arms'][config['arm']]:raise ValueError('balanced batch differs')
    targets=torch.tensor([r['targets'] for r in selected],dtype=torch.long)
    masks=torch.ones_like(targets,dtype=torch.bool);weight=config['auxiliary_weight']
    if (len(selected)!=8 or any(r['supervised']!=[True]*len(r['targets']) for r in selected)
            or not (targets.sum(0)==4).all()):raise ValueError('balanced reviewed supervision differs')
    initial=checkpoint_identity(config['model'])[0]+':'+checkpoint_identity(config['adapter'])[0]
    if report['initial_policy_id']!=initial:raise ValueError('different initial policy')
    reference=json.loads((Path(config['runtime_admission'])/'run/result.json').read_text())
    accepted={r['id']:r for r in reference['records']}
    audits=[]
    for index,row in enumerate(selected):
        other=accepted[row['id']]
        if before['request_ids'][index]!=other['request_id']:
            raise ValueError('changed admitted request identity')
        audits.append(compare_span_policies([p[index] for p in before['candidate_logprobs']],other['candidate_logprobs']))
    if audits!=before['admission'] or not all(a['passed'] for a in audits):
        raise ValueError('actual before-update admission differs')
    for recorded in (before,after):
        expected=expected_rule_reward(tuple(torch.tensor(p) for p in recorded['candidate_logprobs']),targets,masks,weight)
        torch.testing.assert_close(expected,torch.tensor(recorded['expected_reward']),atol=1e-6,rtol=0)
    if config['method']=='rl':
        behavior=torch.load(run/'behavior.pt',map_location='cpu',weights_only=True)
        if behavior['request_ids']!=before['request_ids'] or behavior['policy_id']!=report['initial_policy_id']:
            raise ValueError('behavior identity differs')
        expected_behavior=tuple(torch.tensor([accepted[r['id']]['candidate_logprobs'][j] for r in selected])
                                for j in range(targets.shape[-1]))
        for actual,expected in zip(behavior['candidate_logprobs'],expected_behavior,strict=True):
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        torch.testing.assert_close(joint_logprobs(behavior['candidate_logprobs'],behavior['actions']),behavior['old_joint_logprobs'],atol=0,rtol=0)
        expected=rule_sample_rewards(behavior['actions'],targets,masks,weight)
        for key,value in expected.items():torch.testing.assert_close(value,behavior['rewards'][key],atol=0,rtol=0)
    reload_error=None
    if hf_root is not None:
        from verify_span_rule_baseline import verify as verify_hf
        verified=verify_hf(hf_root)
        if verified['adapter']!=receipt['adapter_id']:raise ValueError('HF reload used another adapter')
        hf={r['id']:r for r in map(json.loads,(hf_root/'run/records.jsonl').read_text().splitlines())}
        actual=torch.tensor([hf[k]['candidate_logprobs'] for k in config['request_ids']])
        expected=torch.tensor(after['candidate_logprobs']).permute(1,0,2)
        reload_error=float((actual-expected).abs().max())
        if reload_error!=0:raise ValueError('independent HF reload differs from post-update probabilities')
    return dict(root=str(root),branch=branch,passed=True,adapter_id=receipt['adapter_id'],
        method=config['method'],auxiliary_weight=weight,optimizer_updates=1,
        mean_expected_reward_before=sum(before['expected_reward'])/8,
        mean_expected_reward_after=sum(after['expected_reward'])/8,
        independent_reload_maximum_error=reload_error,training_admitted=False,
        protocol_sha256=file_hash(root/'protocol.json'),
        scope='Matched learning update/state/reward audit; full version HF/vLLM gate and held-out benefit not asserted')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--branch',required=True);p.add_argument('--hf-root',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=verify(args.root,args.branch,args.hf_root);atomic_json(args.output,result);print(json.dumps(result,indent=2))

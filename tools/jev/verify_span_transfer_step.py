#!/usr/bin/env python3
"""Independent transfer update audit, including optimizer lineage and reload."""
import argparse,json
from pathlib import Path
import torch
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_transfer_runtime import step_rows,cursor,validate_parent
from multimodal_jev.span_training_chain import checkpoint_receipt
from multimodal_jev.span_parity import compare_span_policies
from multimodal_jev.span_rule_objectives import expected_rule_reward,rule_sample_rewards,rule_sft_loss
from multimodal_jev.span_conditional_credit import conditional_credit,reference_conditional_loss
from multimodal_jev.span_policy import joint_logprobs
from multimodal_jev.policy_identity import checkpoint_identity
from verify_span_rule_runtime import verify as verify_runtime
from verify_span_rule_baseline import verify as verify_hf


def verify(root,hf_root=None):
    read=lambda p:json.loads(p.read_text())
    c=read(root/'config.json');run=root/'run';i=c['batch_index']
    for path,digest in c['pins'].items():
        if file_hash(path)!=digest:raise ValueError('training pin changed')
    if read(run/'terminal.json')['exit_code']!=0:raise ValueError('training failed')
    rows=step_rows(c);receipt,_=checkpoint_receipt(run/'policy-1')
    parent=checkpoint_receipt(Path(c['initial_optimizer_checkpoint']))[0] if i else None
    validate_parent(c,parent)
    report=read(run/'update.json');before=read(run/'before.json');after=read(run/'after.json')
    if (receipt['training_state']!=cursor(c,i+1) or receipt['completed_updates']!=i+1
            or report['completed_updates']!=i+1 or report['optimizer_updates']!=1
            or report['optimizer_start']!=('resume' if i else 'fresh')
            or report['exported_policy_id']!=receipt['base_id']+':'+receipt['adapter_id']
            or after['policy_id']!=report['exported_policy_id']
            or report['initial_policy_id']!=receipt['base_id']+':'+c['adapter_id']
            or checkpoint_identity(c['adapter'])[0]!=c['adapter_id']):
        raise ValueError('transfer lineage differs')
    state=torch.load(run/'policy-1/optimizer.pt',map_location='cpu',weights_only=True)
    adam=state['optimizer']['optimizer']['state']
    if (len(adam)!=112 or any(int(s['step'])!=i+1 for s in adam.values())
            or any(v.dtype!=torch.float32 for s in adam.values() for v in s.values() if torch.is_tensor(v))
            or state['training_state']!=receipt['training_state'] or state['adapter_id']!=receipt['adapter_id']):
        raise ValueError('actual optimizer state differs')
    prior=read(run/'optimizer-before.json')
    if prior['optimizer_steps']!=([i] if i else []):raise ValueError('optimizer was not correctly restored')
    runtime_root=Path(c['runtime_admission']);runtime=verify_runtime(runtime_root)
    if not runtime['passed'] or runtime['adapter_id']!=c['adapter_id']:raise ValueError('wrong behavior version')
    accepted={r['id']:r for r in read(runtime_root/'run/result.json')['records']}
    targets=torch.tensor([r['targets'] for r in rows],dtype=torch.long);mask=torch.ones_like(targets,dtype=torch.bool)
    old=tuple(torch.tensor([accepted[r['id']]['candidate_logprobs'][j] for r in rows]) for j in range(targets.shape[1]))
    for n,row in enumerate(rows):
        a=accepted[row['id']]
        if before['request_ids'][n]!=a['request_id']:raise ValueError('request identity differs')
        audit=compare_span_policies([p[n] for p in before['candidate_logprobs']],a['candidate_logprobs'])
        if not audit['passed'] or audit!=before['admission'][n]:raise ValueError('actor gate differs')
    for recorded in (before,after):
        expected=expected_rule_reward(tuple(torch.tensor(p) for p in recorded['candidate_logprobs']),targets,mask,c['auxiliary_weight'])
        torch.testing.assert_close(expected,torch.tensor(recorded['expected_reward']),atol=1e-6,rtol=0)
    actor=tuple(torch.tensor(p) for p in before['candidate_logprobs'])
    if c['method']=='conditional':
        b=torch.load(run/'conditional-credit.pt',map_location='cpu',weights_only=True)
        adv,prob=conditional_credit(old,targets,mask,c['auxiliary_weight'])
        for k,v in [('advantages',adv),('probabilities',prob),('targets',targets),('supervised',mask)]:
            torch.testing.assert_close(b[k],v,atol=0,rtol=0)
        expected=reference_conditional_loss(old,actor,targets,mask,c['auxiliary_weight'])
        if abs(float(expected)-report['loss'])>1e-6:raise ValueError('conditional loss differs')
    elif c['method']=='rl':
        b=torch.load(run/'behavior.pt',map_location='cpu',weights_only=True)
        torch.testing.assert_close(joint_logprobs(old,b['actions']),b['old_joint_logprobs'],atol=0,rtol=0)
        rewards=rule_sample_rewards(b['actions'],targets,mask,c['auxiliary_weight'])
        for k,v in rewards.items():torch.testing.assert_close(v,b['rewards'][k],atol=0,rtol=0)
    else:
        b=None
        loss=rule_sft_loss(actor,targets,mask,c['auxiliary_weight'])
        if abs(float(loss)-report['loss'])>1e-6:raise ValueError('SFT loss differs')
    if b is not None:
        if b['request_ids']!=before['request_ids'] or b['policy_id']!=report['initial_policy_id']:raise ValueError('behavior identity differs')
        for x,y in zip(old,b['candidate_logprobs'],strict=True):torch.testing.assert_close(x,y,atol=0,rtol=0)
    error=None
    if hf_root:
        hf=verify_hf(hf_root)
        if hf['adapter']!=receipt['adapter_id']:raise ValueError('independent HF adapter differs')
        predictions={r['id']:r for r in map(json.loads,(hf_root/'run/records.jsonl').read_text().splitlines())}
        observed=torch.tensor([predictions[r['id']]['candidate_logprobs'] for r in rows])
        expected=torch.tensor(after['candidate_logprobs']).permute(1,0,2)
        error=float((observed-expected).abs().max())
        if error!=0:raise ValueError('independent HF reload differs')
    return dict(passed=True,completed_updates=i+1,adapter_id=receipt['adapter_id'],
        independent_reload_maximum_error=error,protocol_sha256=file_hash(c['transfer_protocol']),
        optimizer_sha256=receipt['optimizer_sha256'],scope='One transfer update and actual resume lineage; next runtime version needs separate acceptance')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--hf-root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=verify(a.root,a.hf_root);atomic_json(a.output,r);print(json.dumps(r,indent=2))

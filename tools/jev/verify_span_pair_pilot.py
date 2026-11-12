#!/usr/bin/env python3
"""Independent one-step audit; only full post-export gates permit acceptance."""
import argparse
import json
from pathlib import Path
import torch
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_pair_pilot import pilot_rows
from multimodal_jev.span_pair_credit import exact_pair_credit,expected_pair_reward
from multimodal_jev.span_pair_verl import reference_pair_loss
from multimodal_jev.span_parity import compare_span_policies
from multimodal_jev.span_training_chain import checkpoint_receipt
from verify_span_rule_runtime import verify as verify_runtime
from verify_span_rule_baseline import verify as verify_hf


def audit_payload(config,protocol,batch,rows,pairs,before,after,update,saved,checkpoint,optimizer,accepted):
    targets=torch.tensor([r['targets'][-1] for r in rows],dtype=torch.long)
    old=tuple(torch.tensor([accepted[r['id']]['candidate_logprobs'][j] for r in rows]) for j in range(3))
    ids=[accepted[r['id']]['request_id'] for r in rows];beta=config['pair_weight']
    initial=protocol['base_id']+':'+protocol['initial_adapter_id']
    exported=checkpoint['base_id']+':'+checkpoint['adapter_id']
    expected_state=dict(schema='jev-pair-engineering-pilot-v1',protocol_sha256=file_hash(config['protocol']),pair_weight=beta,completed_updates=1)
    if (before['request_ids']!=ids or saved['request_ids']!=ids or saved['policy_id']!=initial
            or update['initial_policy_id']!=initial or update['exported_policy_id']!=exported
            or after['policy_id']!=exported or checkpoint['base_id']!=protocol['base_id']
            or checkpoint['completed_updates']!=1 or checkpoint['training_state']!=expected_state
            or update['optimizer_updates']!=1 or update['training_requests']!=8 or update['pairs']!=4
            or update['pair_weight']!=beta or update['auxiliary_weight']!=0.
            or update['algorithm']!=protocol['algorithm'] or update['checkpoint_ready'] is not False
            or update['training_admitted'] is not False):
        raise ValueError('paired update identity or contract differs')
    for key,value in [('targets',targets),('pairs',pairs),('advantages',exact_pair_credit(old[-1],targets,pairs,beta))]:
        torch.testing.assert_close(saved[key],value,atol=0,rtol=0)
    for actual,expected in zip(saved['candidate_logprobs'],old,strict=True):
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    actor=tuple(torch.tensor(p) for p in before['candidate_logprobs'])
    audits=[compare_span_policies([p[i].tolist() for p in actor],[p[i].tolist() for p in old]) for i in range(8)]
    if not all(a['passed'] for a in audits) or audits!=update['admission']:
        raise ValueError('full candidate pre-update admission differs')
    loss=reference_pair_loss(old[-1],actor[-1],targets,pairs,beta)
    if abs(float(loss)-update['loss'])>1e-6:raise ValueError('paired loss differs')
    expected=expected_pair_reward(old[-1],targets,pairs,beta).mean()
    if abs(float(expected)-update['expected_reward_before'])>1e-6:raise ValueError('pre-update reward differs')
    probability=torch.tensor(after['candidate_logprobs'][-1])
    torch.testing.assert_close(expected_pair_reward(probability,targets,pairs,beta),torch.tensor(after['expected_pair_reward']),atol=1e-6,rtol=0)
    adam=optimizer['optimizer']['optimizer']['state'];masters=optimizer['optimizer']['masters']
    if (optimizer['base_id']!=checkpoint['base_id'] or optimizer['adapter_id']!=checkpoint['adapter_id']
            or optimizer['completed_updates']!=1 or optimizer['training_state']!=expected_state
            or len(adam)!=112 or len(masters)!=112 or len(set(optimizer['ordered_parameter_names']))!=112
            or any(int(s['step'])!=1 for s in adam.values())
            or any(v.dtype!=torch.float32 or not torch.isfinite(v).all()
                   for s in adam.values() for v in s.values() if torch.is_tensor(v))
            or any(m.dtype!=torch.float32 or not torch.isfinite(m).all() for m in masters)):
        raise ValueError('actual paired optimizer state differs')
    return dict(passed=True,adapter_id=checkpoint['adapter_id'],pair_weight=beta,optimizer_updates=1)


def verify(stage,hf_root=None,runtime_root=None):
    read=lambda p:json.loads(p.read_text())
    config=read(stage/'config.json');run=stage/'run'
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('paired input pin changed')
    protocol,batch,rows,pairs=pilot_rows(config)
    if read(run/'terminal.json')['exit_code']!=0:raise ValueError('paired training failed')
    initial=verify_runtime(Path(protocol['runtime_admission']))
    if not initial['passed'] or initial['adapter_id']!=protocol['initial_adapter_id'] or initial['base_id']!=protocol['base_id']:
        raise ValueError('initial runtime failed')
    accepted={r['id']:r for r in read(Path(protocol['runtime_admission'])/'run/result.json')['records']}
    checkpoint,_=checkpoint_receipt(run/'policy-1')
    saved=torch.load(run/'pair-credit.pt',map_location='cpu',weights_only=True)
    optimizer=torch.load(run/'policy-1/optimizer.pt',map_location='cpu',weights_only=True)
    before,after,update=[read(run/(n+'.json')) for n in ('before','after','update')]
    report=audit_payload(config,protocol,batch,rows,pairs,before,after,update,saved,checkpoint,optimizer,accepted)
    if read(run/'optimizer-before.json')['optimizer_steps']!=[]:raise ValueError('fresh optimizer required')
    report.update(accepted_for_reuse=False,independent_reload_maximum_error=None)
    if (hf_root is None)!=(runtime_root is None):raise ValueError('both independent gates required')
    if hf_root is not None:
        hf=verify_hf(hf_root);runtime=verify_runtime(runtime_root)
        original=read(Path(protocol['runtime_admission'])/'config.json')
        for root in (hf_root,runtime_root):
            c=read(root/'config.json')
            if c['request_ids']!=original['request_ids'] or c['release']!=batch['release']:
                raise ValueError('complete original runtime coverage required')
        if (not hf['passed'] or not runtime['passed'] or hf['adapter']!=checkpoint['adapter_id']
                or runtime['adapter_id']!=checkpoint['adapter_id']
                or Path(read(runtime_root/'config.json')['hf_run']).resolve()!=hf_root.resolve()):
            raise ValueError('post-update policy gate differs')
        predictions={r['id']:r for r in map(json.loads,(hf_root/'run/records.jsonl').read_text().splitlines())}
        observed=torch.tensor([predictions[r['id']]['candidate_logprobs'] for r in rows])
        expected=torch.tensor(after['candidate_logprobs']).permute(1,0,2)
        error=float((observed-expected).abs().max())
        if error!=0:raise ValueError('independent HF reload differs')
        report.update(accepted_for_reuse=True,independent_reload_maximum_error=error,runtime=runtime)
    report.update(files={str(run/n):file_hash(run/n) for n in ('before.json','after.json','update.json','pair-credit.pt','terminal.json')},
        scope='One independent paired engineering update; full gates required for reuse. No task benefit or continuation admission.')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hf-root',type=Path);p.add_argument('--runtime-root',type=Path);a=p.parse_args()
    r=verify(a.stage,a.hf_root,a.runtime_root);atomic_json(a.output,r);print(json.dumps(r,indent=2))

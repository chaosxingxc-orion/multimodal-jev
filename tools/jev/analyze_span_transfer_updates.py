#!/usr/bin/env python3
"""Reaudit a completed chain and decompose actual same-batch policy changes."""
import argparse
import json
from pathlib import Path
import torch

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.span_transfer_runtime import step_rows
from multimodal_jev.span_update_diagnostic import diagnose_update
from verify_span_transfer_chain import verify


def analyze(root):
    audit = verify(root)
    config = json.loads((root/'config.json').read_text())
    results = []
    files = {}
    for index in range(audit['initial_completed_updates']+1,audit['completed_updates']+1):
        stage = root/'run'/f'update-{index:04d}'
        c = json.loads((stage/'config.json').read_text())
        rows = step_rows(c)
        target = torch.tensor([r['targets'] for r in rows],dtype=torch.long)
        mask = torch.tensor([r['supervised'] for r in rows],dtype=torch.bool)
        saved = {}
        for name in ('before','after','update'):
            path = stage/'run'/(name+'.json')
            saved[name] = json.loads(path.read_text())
            files[str(path)] = file_hash(path)
        files[str(stage/'config.json')] = file_hash(stage/'config.json')
        runtime_path = Path(c['runtime_admission'])/'run/result.json'
        admitted = {r['id']:r['request_id'] for r in json.loads(runtime_path.read_text())['records']}
        files[str(runtime_path)] = file_hash(runtime_path)
        if saved['before']['request_ids'] != [admitted[r['id']] for r in rows]:
            raise ValueError('training request order differs')
        old,new = [tuple(torch.tensor(p,dtype=torch.float64) for p in saved[name]['candidate_logprobs'])
                   for name in ('before','after')]
        report = diagnose_update(old,new,target,mask,c['auxiliary_weight'])
        metrics = saved['update'].get('per_request_verl_metrics',[])
        report.update(update=index, gradient_norm=saved['update']['gradient_norm'],
            reported_loss=saved['update']['loss'],
            maximum_reported_clipfrac=max((m['actor/pg_clipfrac'] for m in metrics),default=None))
        results.append(report)
    return dict(passed=True,training_root=str(root),chain_audit=audit,files=files,
        method=config['method'],seed=config['seed'],updates=results,
        positive_same_batch_reward_steps=sum(r['reward_delta']>0 for r in results),
        negative_same_batch_reward_steps=sum(r['reward_delta']<0 for r in results),
        scope='Post-hoc learning-batch diagnostic, not population gain or a selection metric. '
              'Different steps use different batches; their deltas must not be summed as trajectory gain. '
              'Logit-space linearization cannot identify shared-parameter interference or Adam causality. '
              'No changed reward, optimizer, checkpoint selection or inference.')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=analyze(a.root);atomic_json(a.output,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('chain_audit','files','updates')},indent=2))
    for row in report['updates']:
        print(json.dumps({k:row[k] for k in ('update','reward_delta','linearized_reward_delta','nonlinear_remainder')}))

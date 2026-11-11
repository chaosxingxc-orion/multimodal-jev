#!/usr/bin/env python3
"""Authenticate fixed-initial/final inputs and report all transfer cells."""
import argparse,json
from copy import deepcopy
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_rule_population import configured_rule_rows
from multimodal_jev.span_transfer_evaluation import cell_metrics
from verify_span_rule_baseline import verify as verify_hf
from verify_span_transfer_chain import verify as verify_chain


def compare(root):
    read=lambda p:json.loads(p.read_text());protocol=read(root/'evaluation-protocol.json');study=read(root/'study.json')
    if read(root/'terminal.json')['exit_code']!=0:raise ValueError('complete evaluation required')
    chain=verify_chain(Path(protocol['training_root']))
    if chain['completed_updates']!=study['updates_per_branch']:raise ValueError('fixed final update required')
    rows=[];old=[];new=[];audits={}
    for role in ('learning','selection'):
        records={}
        for policy in ('initial','final'):
            stage=root/(policy+'-'+role);audits[stage.name]=verify_hf(stage)
            c=read(stage/'config.json')
            expected=study['initializer']['adapter_id'] if policy=='initial' else chain['versions'][-1]['adapter_id']
            if (c['adapter_id']!=expected or c['evaluation_role']!=role
                    or file_hash(Path(c['evaluation_protocol']))!=file_hash(root/'evaluation-protocol.json')
                    or file_hash(Path(c['transfer_protocol']))!=file_hash(root/'study.json')):
                raise ValueError('evaluation stage identity differs')
            records[policy]=[json.loads(s) for s in (stage/'run/records.jsonl').read_text().splitlines()]
            if policy=='initial':rows.extend(configured_rule_rows(c))
        for a,b in zip(records['initial'],records['final'],strict=True):
            x,y=deepcopy(a['request_identity']),deepcopy(b['request_identity']);x.pop('policy_id');y.pop('policy_id')
            if a['id']!=b['id'] or x!=y:raise ValueError('cross-policy inputs differ')
        old.extend(records['initial']);new.extend(records['final'])
    return dict(passed=True,training=chain,study_sha256=file_hash(root/'study.json'),
        evaluation_protocol_sha256=file_hash(root/'evaluation-protocol.json'),audits=audits,
        cells=cell_metrics(study,rows,old,new),
        scope='Fixed terminal policy versus initializer; one seed and algorithm, within-SLURP source/rule transfer development diagnostics, not causal RL advantage or cross-dataset generalization')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=compare(a.root);atomic_json(a.output,r);print(json.dumps({k:v for k,v in r.items() if k not in ('training','audits','cells')},indent=2))

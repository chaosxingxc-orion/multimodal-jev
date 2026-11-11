#!/usr/bin/env python3
"""Authenticate frozen outputs and recompute rule/control metrics offline."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.span_rule_metrics import score_rules
from infer_span_rule_baseline import select_rows


def verify(root):
    read=lambda p:json.loads(p.read_text())
    config=read(root/'config.json');launch=read(root/'launch.json');run=root/'run'
    if (launch['physical_devices']!=[5] or launch['config_sha256']!=file_hash(root/'config.json')
            or launch['source_manifest_sha256']!=file_hash(root/'source-manifest.json')
            or read(run/'terminal.json')['exit_code']!=0):raise ValueError('launch or terminal differs')
    for p,d in config['pins'].items():
        if file_hash(p)!=d:raise ValueError('input pin differs')
    result=read(run/'result.json')
    adapter_id=checkpoint_identity(config['adapter'])[0] if config.get('adapter') else None
    if (result['base_id']!=config['base_id'] or checkpoint_identity(config['model'])[0]!=config['base_id']
            or result['adapter']!=adapter_id or config.get('adapter_id')!=adapter_id):
        raise ValueError('immutable policy differs')
    if file_hash(run/'records.jsonl')!=result['records_sha256']:raise ValueError('predictions changed')
    rows=select_rows(Path(config['release']))
    records=[json.loads(s) for s in (run/'records.jsonl').read_text().splitlines()]
    if [r['id'] for r in rows]!=config['request_ids'] or [r['id'] for r in records]!=config['request_ids']:
        raise ValueError('request coverage differs')
    for row,record in zip(rows,records,strict=True):
        if len(record['candidate_logprobs'])!=len(row['targets']):raise ValueError('span width differs')
        predicted=[]
        for p in record['candidate_logprobs']:
            if len(p)!=2 or not all(math.isfinite(v) for v in p) or abs(sum(math.exp(v) for v in p)-1)>1e-6:
                raise ValueError('invalid categorical probabilities')
            predicted.append(p[1]>p[0])
        if predicted!=record['predictions']:raise ValueError('argmax output differs')
        identity=record['request_identity']
        # The request payload is independently bound by the same canonical identity protocol.
        digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if digest!=record['request_id']:raise ValueError('request digest differs')
        if identity['request']!=row['request'] or identity['policy_id']!=config['base_id']+':'+(adapter_id or 'no-adapter'):
            raise ValueError('recorded input or policy differs')
        if file_hash(row['request']['audio'])!=row['provenance']['audio_sha256']:
            raise ValueError('source audio changed')
    metrics=score_rules(rows,records)
    return dict(root=str(root),passed=True,base_id=result['base_id'],adapter=adapter_id,
        requests=len(records),recordings=result['recordings'],metrics=metrics,
        files={str(p):file_hash(p) for p in [root/'config.json',run/'result.json',run/'records.jsonl',run/'terminal.json']},
        training_admitted=False,scope='Frozen HF audio Span learning-only baseline; no cross-backend, task-generalization or RL-gain claim')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=verify(args.root);atomic_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','metrics')},indent=2))
    print(json.dumps({k:v for k,v in result['metrics'].items() if k!='per_rule'},indent=2))

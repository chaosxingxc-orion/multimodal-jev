#!/usr/bin/env python3
"""Evaluate a complete authenticated split from context engine records."""
import argparse,json
from pathlib import Path
from multimodal_jev.baseline import file_hash
from multimodal_jev.context_event_metrics import ContextEventMetrics
from multimodal_jev.slurp_portable_release import load


def evaluate(release,release_sha256,predictions,predictions_sha256,policy_id,role):
    if role!='selection':raise ValueError('selection-only evaluation; training diagnostics are separate')
    if file_hash(predictions)!=predictions_sha256:raise ValueError('prediction file changed')
    rows,task,_=load(release,release_sha256,role,purpose='evaluate')
    source={r['id']:r for r in rows}
    data=json.loads(Path(predictions).read_text());records=data['records']
    if len(records)!=len(source) or {r['id'] for r in records}!=set(source):
        raise ValueError('complete unique split predictions required; subsets cannot report split metrics')
    metric=ContextEventMetrics([c['id'] for c in task['schema']['candidates']],threshold=.5)
    for row in records:
        if row['policy_id']!=policy_id or row['request_identity']['policy_id']!=policy_id:
            raise ValueError('mixed or stale prediction policy')
        identity=row['request_identity']
        import hashlib
        rid=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if row['request_id']!=rid or len(identity['positions'])!=60:raise ValueError('engine input binding differs')
        metric.add(row['id'],source[row['id']]['intent'],row['prob'])
    return dict(policy_id=policy_id,role=role,release_sha256=release_sha256,
        predictions_sha256=predictions_sha256,metrics=metric.result(),
        scope='Complete published split IDs and engine-record identities scored; caller must independently authenticate audio/request tensors and policy assets.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('release','release-sha256','predictions','predictions-sha256','policy-id','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--role',choices=['selection'],required=True);a=p.parse_args()
    result=evaluate(a.release,a.release_sha256,a.predictions,a.predictions_sha256,a.policy_id,a.role)
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')

"""Deterministic equality and throughput report from independently audited runs."""
import argparse
from datetime import datetime
import json
from pathlib import Path

from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from tools.reports.prepare_objective_training_candidate import read_bound
from tools.reports import context_replica_selection_operations as old
from tools.reports import multidevice_selection_operations as multi


def compare(left,right,*,left_seconds,right_seconds):
    a=left['records'];b=right['records']
    if not a or len(a)!=len(b) or len({r['id'] for r in a})!=len(a):
        raise ValueError('same complete unique populations required')
    keys=('id','policy_id','request_id','request_identity','prob','logits')
    if any(any(x[k]!=y[k] for k in keys) for x,y in zip(a,b,strict=True)):
        raise ValueError('per-example output or policy differs')
    import math
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (left_seconds,right_seconds)):
        raise ValueError('positive complete inference durations required')
    return dict(recordings=len(a),all_logits_and_probabilities_exact=True,
        left_queries_per_second=len(a)/left_seconds,right_queries_per_second=len(b)/right_seconds,
        inference_speedup=left_seconds/right_seconds)


def run(c,provenance):
    values=[];audits=[];durations=[]
    for entry in c['runs']:
        bound=json.loads(read_bound(entry['audit_config']))
        if entry['kind'] not in ('legacy','multidevice'):raise ValueError('known audited evaluation kind required')
        api=old if entry['kind']=='legacy' else multi
        proof=api.audit(bound,provenance);audits.append(proof)
        if proof['status']!='complete_replica_infonce_selection_audited':raise ValueError('complete point required')
        root=Path(bound['collection'])/'evaluation/inference'
        values.append((json.loads((root/'predictions.json').read_text()),json.loads((root/'result.json').read_text())))
        state=json.loads(Path(bound['inspect']).read_text())['State']
        # Docker timestamps include nanoseconds; fromisoformat accepts these.
        duration=(datetime.fromisoformat(state['FinishedAt'].replace('Z','+00:00'))-
                  datetime.fromisoformat(state['StartedAt'].replace('Z','+00:00'))).total_seconds()
        durations.append(duration)
    if len(values)!=2:raise ValueError('exactly two matched configurations required')
    for key in ('policy_id','role','release_sha256','metrics'):
        if audits[0]['result'][key]!=audits[1]['result'][key]:raise ValueError('full native metrics differ: '+key)
    # The exact scorer results bind dataset release and policy, in addition to
    # per-row inputs. Runtime equality is checked explicitly here.
    if values[0][1]['runtime_identity']!=values[1][1]['runtime_identity']:
        raise ValueError('same forward runtime required')
    result=compare(values[0][0],values[1][0],left_seconds=values[0][1]['seconds'],right_seconds=values[1][1]['seconds'])
    result.update(status='matched_multidevice_evaluation_compared',container_seconds=durations,
        container_speedup=durations[0]/durations[1],metrics_exact=True,
        timing_caveat='Container time excludes preparation/collection/audit; historical runs do not control cache/host load',
        provenance=provenance)
    commit_json(c['output'],result);return result


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local);print(json.dumps(run(c,provenance),indent=2))

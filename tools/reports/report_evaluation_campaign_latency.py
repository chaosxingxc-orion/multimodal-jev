"""Reconstruct measured legacy latency; unknown phases remain unknown."""
import argparse
from datetime import datetime
import json
from pathlib import Path

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config


def report(collection, inspect_path):
    root=Path(collection)
    def read(p):return json.loads(Path(p).read_text())
    state=read(inspect_path)['State']
    wall=(datetime.fromisoformat(state['FinishedAt'].replace('Z','+00:00'))-
          datetime.fromisoformat(state['StartedAt'].replace('Z','+00:00'))).total_seconds()
    inference=read(root/'evaluation/inference/result.json')
    total=critical=balanced=0
    for batch in inference['batches']:
        value=read(root/'evaluation/inference'/batch['path']);receipt=value['macro_receipt']
        total+=receipt['seconds'];loads=[0.]*4
        journal=root/'selection-journal/selection-v1'/receipt['batch_id']
        for p in journal.glob('rollout-*.json'):
            row=read(p);loads[row['replica_id']]+=row['seconds']
        critical+=max(loads);balanced+=sum(loads)/4
    return dict(scope='one completed checkpoint; not a complete campaign baseline',
        recordings=inference['recordings'],policy_id=inference['policy_id'],container_seconds=wall,
        client_seconds=inference['seconds'],macro_seconds=total,
        client_outside_macros_seconds=inference['seconds']-total,
        container_outside_client_seconds=wall-inference['seconds'],
        service_critical_path_sum_seconds=critical,perfect_balance_proxy_seconds=balanced,
        unknown_separate_phases=['controller_preparation','input_load','hf_panel','engine_start',
                                'output_write','collection','cpu_audit'],
        caution='Nested timings are not additive; balance proxy is not a measured optimization',
        formal_admitted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    result=report(c['collection'],c['inspect']);commit_json(c['output'],dict(result,provenance=provenance))
    print(json.dumps(result,indent=2))

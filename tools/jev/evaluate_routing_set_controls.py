#!/usr/bin/env python3
"""Full-learning diagnostic controls, not predictions or held-out results."""
import argparse
import json
from pathlib import Path

from audit_harness_routing_support import FAMILIES, policy, target
from multimodal_jev.baseline import file_hash
from multimodal_jev.candidate_set_metrics import CandidateSetMetrics
from multimodal_jev.slurp_official_training import load
from multimodal_jev.slurp_training import content_hash


def evaluate(rows, task):
    families=[]
    ontology=sorted(task['candidate_pairs'])
    for family in FAMILIES:
        spec=policy(ontology,family,317)
        controls={name:CandidateSetMetrics(.5) for name in
                  ('return_all','return_none','return_enabled','first_enabled','oracle_set','oracle_one')}
        for row in rows:
            truth=target(row['intent'],ontology,spec)['candidate_applicability']
            ids=list(truth)
            enabled=[r['id'] for r in spec['routes'] if r['enabled']]
            valid=[k for k,v in truth.items() if v]
            selected=dict(return_all=set(ids),return_none=set(),return_enabled=set(enabled),
                          first_enabled=set(enabled[:1]),oracle_set=set(valid),oracle_one=set(valid[:1]))
            for name,values in selected.items():
                controls[name].add(row['id'],truth,{k:float(k in values) for k in ids})
        families.append(dict(family=family,policy_sha256=content_hash(spec),
                             controls={k:v.result() for k,v in controls.items()}))
    return dict(schema='harness-routing-set-controls-v1',recordings=len(rows),
        source_groups=len({r['source_group'] for r in rows}),seed=317,
        population_sha256=content_hash(sorted((r['id'],r['source_group'],r['intent'],r['audio_sha256']) for r in rows)),
        policy_families=families,model_predictions=False,training_dispatch_allowed=False,
        scope='Authenticated learning-population diagnostic controls. oracle_set/oracle_one use gold only as metric controls, never deployable policies. Endpoint probabilities are deterministic controls, not calibrated model estimates. No new semantic supervision or test performance.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--release-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows,task,_=load(a.release,a.release_sha256,'learning',purpose='optimize')
    result=evaluate(rows,task)
    result.update(release=str(a.release.resolve()),release_sha256=a.release_sha256,
                  code_sha256={path:file_hash(path) for path in [__file__,
                    'tools/jev/audit_harness_routing_support.py','src/multimodal_jev/candidate_set_metrics.py']})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps([dict(family=f['family'],controls={k:{m:v[m] for m in
          ['micro_precision_observed','micro_recall_observed','exact_set_accuracy_complete','mean_returned_candidates']}
          for k,v in f['controls'].items()}) for f in result['policy_families']],indent=2))

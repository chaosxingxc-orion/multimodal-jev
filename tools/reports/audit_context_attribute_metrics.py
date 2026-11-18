"""Full held-out population synthetic score controls, never model evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from tools.reports.context_attribute_metrics_candidate import score_records


def run(root):
    root=Path(root)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(root/'MANIFEST.json')!='85d17ce7a12d2e39efb9bdc676cdb24b02f2531ccb4c93b7adeba303485d0710':
        raise ValueError('Frozen held-out candidate required')
    manifest=json.loads((root/'MANIFEST.json').read_text())
    for name,h in manifest['files'].items():
        if sha(root/name)!=h:raise ValueError('Candidate changed')
    rows=lambda n:[json.loads(s) for s in (root/n).read_text().splitlines()]
    inputs=rows('inputs.jsonl');targets=rows('targets.jsonl');menu=json.loads((root/'catalog.json').read_text())
    ids=[n['id'] for n in menu['nodes']];controls={}
    for control in ('perfect','empty','all','process_only'):
        predictions=[]
        for inp,t in zip(inputs,targets,strict=True):
            if control=='perfect':p=[.9 if y else .1 for y in t['labels']]
            elif control=='empty':p=[.1]*124
            elif control=='all':p=[.9]*124
            else:p=[.9 if y and not k.startswith('final:') else .1 for k,y in zip(ids,t['labels'],strict=True)]
            predictions.append(dict(id=inp['id'],arm='q_credit',policy_id='synthetic-control',decision_ids=ids,prob=p))
        result=score_records(inputs,targets,predictions,menu,arm='q_credit',policy_id='synthetic-control',expected_count=9226)
        for family,n in [('final',60),('scenario',18),('action',46)]:
            r=result['metrics'][family]
            perfect=control=='perfect' or control=='process_only' and family!='final'
            expected_f1=1. if perfect else 2/(n+1) if control=='all' else 0.
            expected_nll=-math.log(.9) if perfect else -(math.log(.9)+(n-1)*math.log(.1))/n if control=='all' else -(math.log(.1)+(n-1)*math.log(.9))/n
            assert abs(r['query_macro_set_f1']-expected_f1)<1e-12
            assert abs(r['candidate_binary_nll']-expected_nll)<1e-11
            assert r['queries']==9226 and sum(v[family]['queries'] for v in result['structural_slices'].values())==9226
        assert result['source_groups']==2146
        controls[control]=dict(metrics=result['metrics'],structural_slices=result['structural_slices'])
    return dict(passed=True,recordings=9226,source_groups=2146,controls=controls,
        manifest_sha256=sha(root/'MANIFEST.json'),
        scope='Full selection artificial probability controls with analytic F1/NLL expectations; no model inference or efficacy result')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=run(a.dataset)
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')

#!/usr/bin/env python3
"""Check equal-input/equal-schedule candidate-vs-full CE learning diagnostics."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash

p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
left,right=a.baseline,a.candidate
configs=[json.loads((root/'config.json').read_text()) for root in (left,right)]
assert configs[0].get('objective','full_vocabulary_ce')=='full_vocabulary_ce'
assert configs[1]['objective']=='candidate_ce'
assert {k:v for k,v in configs[0].items() if k not in {'pins','objective'}}=={k:v for k,v in configs[1].items() if k not in {'pins','objective'}}
identities=[json.loads((root/'run/identity.json').read_text()) for root in (left,right)]
assert identities[0]['bundles']==identities[1]['bundles']
assert identities[0]['model_weights_sha256']==identities[1]['model_weights_sha256']
balance=[json.loads((root/'run/balance.json').read_text()) for root in (left,right)]
assert balance[0]==balance[1]
histories=[json.loads((root/'run/history.json').read_text()) for root in (left,right)]
assert [h['recordings'] for h in histories[0]]==[h['recordings'] for h in histories[1]]
assert [h['supervised_spans'] for h in histories[0]]==[h['supervised_spans'] for h in histories[1]]
assert json.loads((left/'run/before.json').read_text())==json.loads((right/'run/before.json').read_text())
results={}
for phase in ('after','after-silence'):
    l=json.loads((left/'run'/f'{phase}.json').read_text());r=json.loads((right/'run'/f'{phase}.json').read_text())
    fixed=broken=0
    for old,new in zip(l['records'],r['records'],strict=True):
        assert old['id']==new['id']
        for x,y in zip(old['correct'],new['correct'],strict=True):
            fixed+=x is False and y is True
            broken+=x is True and y is False
    assert r['correct']-l['correct']==fixed-broken
    results[phase]=dict(baseline_correct=l['correct'],candidate_correct=r['correct'],
                       wrong_to_right=fixed,right_to_wrong=broken,denominator=r['decisions'],
                       baseline_complete=l['complete_correct'],candidate_complete=r['complete_correct'],
                       candidate_per_span=r['per_span'])
atomic_json(a.output,dict(status='passed',baseline=str(left),candidate=str(right),
    checks=['same hyperparameters except objective','identical input token/positions/audio receipt and targets',
            'same base weight hash','same 40 recordings and 20 update schedule','identical before probabilities',
            'paired correctness recomputed'],results=results,
    files={str(root/name):file_hash(root/name) for root in (left,right) for name in ('config.json','run/result.json','run/history.json','run/identity.json','run/before.json','run/after.json','run/after-silence.json')},
    scope='Fixed-seed learning-population objective comparison, not held-out or CoT/RL improvement.'))
print(json.dumps(results,indent=2))

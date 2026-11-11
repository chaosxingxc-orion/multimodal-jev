#!/usr/bin/env python3
"""Independently recompute balanced run schedule and all diagnostic confusion tables."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import file_hash,atomic_json
from multimodal_jev.span_balance import balanced_binary_batches
from multimodal_jev.span_training import load_reviewed_bundles

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();root=a.root
c=json.loads((root/'config.json').read_text());r=json.loads((root/'run/result.json').read_text());h=json.loads((root/'run/history.json').read_text())
assert json.loads((root/'run/terminal.json').read_text())['exit_code']==0
bundles,_=load_reviewed_bundles(c['release'],c['manifest_sha256'],c['question_order'])
schedule,balance=balanced_binary_batches(bundles,c['seed'])
assert balance==json.loads((root/'run/balance.json').read_text())
assert [x['recordings'] for x in h]==balance['batch_recording_ids']
assert len(h)==20 and sum(x['supervised_spans'] for x in h)==120
assert all(all(sum(bundles[i]['targets'][j] for i in batch)==1 for j in range(3)) for batch in schedule)
for name,digest in r['adapter_files'].items():assert file_hash(root/'run/adapter'/name)==digest
summaries={}
for phase in ['before','after','after-silence']:
    saved=json.loads((root/'run'/f'{phase}.json').read_text())
    assert [x['id'] for x in saved['records']]==[b['id'] for b in bundles]
    tables=[[[0,0],[0,0]] for _ in c['question_order']]
    correct=0
    for b,row in zip(bundles,saved['records']):
        checks=[pred==gold if active else None for pred,gold,active in zip(row['predictions'],b['targets'],b['supervised'])]
        assert checks==row['correct']
        correct+=sum(x is True for x in checks)
        for j,active in enumerate(b['supervised']):
            if active:tables[j][int(b['targets'][j])][int(row['predictions'][j])]+=1
    for j,name in enumerate(c['question_order']):
        assert tables[j]==saved['per_span'][name]['confusion_gold_by_prediction']
    assert correct==saved['correct']
    summary={k:v for k,v in saved.items() if k!='records'}
    assert summary==r[phase.replace('-','_')]
    summaries[phase]=summary
receipt={'status':'passed','root':str(root),'verified_updates':len(h),'training_recordings':40,'supervised_spans':120,'every_batch_each_task_positive_negative':[1,1],'metrics':summaries,'files':{n:file_hash(root/n) for n in ['config.json','run/result.json','run/history.json','run/balance.json','run/before.json','run/after.json','run/after-silence.json','run/terminal.json']},'scope':'Schedule and per-recording metrics independently recomputed; no reload/held-out claim.'}
atomic_json(a.output,receipt)
print(json.dumps({'status':'passed','metrics':{k:v['correct'] for k,v in summaries.items()}}))

#!/usr/bin/env python3
"""Score completed full-valid baseline predictions with macro/per-class recall."""
import argparse,json
from collections import Counter
from pathlib import Path
from multimodal_jev.baseline import file_hash


def summarize(run):
    result=json.loads((run/'result.json').read_text());identity=result['identity']
    manifest=Path(identity['config']['valid_manifest'])
    if file_hash(manifest)!=identity['valid_manifest_sha256']:raise ValueError('manifest changed')
    targets={}
    for line in manifest.open():
        record=json.loads(line)
        if record['metadata']['variant']=='native':targets[record['example_id']]=record['target']
    support=Counter(targets.values());scores=[]
    for seed in identity['config']['seeds']:
        path=run/f'valid-predictions-{seed}.jsonl'
        predictions={}
        for line in path.open():
            row=json.loads(line)
            if row['example_id'] in predictions:raise ValueError('duplicate prediction')
            predictions[row['example_id']]=row['value']
        if set(predictions)!=set(targets):raise ValueError('full official valid prediction coverage required')
        correct=Counter(targets[key] for key,value in predictions.items() if targets[key]==value)
        scores.append({'seed':seed,'accuracy':sum(correct.values())/len(targets),
                       'macro_recall':sum(correct[label]/n for label,n in support.items())/len(support),
                       'per_class':{label:{'correct':correct[label],'total':n,'recall':correct[label]/n} for label,n in sorted(support.items())},
                       'predictions_sha256':file_hash(path)})
    return {'status':'passed','valid_recordings':len(targets),'classes':len(support),
            'majority_class_accuracy':max(support.values())/len(targets),'seeds':scores,
            'result_sha256':file_hash(run/'result.json'),'scope':'full official FSC valid closed-label control; test untouched'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=summarize(args.run)
    with args.output.open('x') as handle:json.dump(result,handle,indent=2)

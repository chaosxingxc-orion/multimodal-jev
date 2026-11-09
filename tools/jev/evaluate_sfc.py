#!/usr/bin/env python3
"""Pair unchanged official SFC metrics with strict metrics on aligned raw outputs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.sfc import strict_metrics


def evaluate(manifest,predictions,official_scorer,output,expected_scorer_hash):
    if file_hash(official_scorer)!=expected_scorer_hash:
        raise ValueError('official scorer differs from audited version')
    rows=[json.loads(line) for line in manifest.open()]
    by_id={r['example_id']:r for r in rows}
    if len(by_id)!=len(rows) or not rows:raise ValueError('nonempty unique manifest required')
    splits={r['metadata']['official_split'] for r in rows}
    if len(splits)!=1:raise ValueError('score official splits separately')
    predicted={}
    for line in predictions.open():
        row=json.loads(line);key=row['example_id']
        if key in predicted or key not in by_id:raise ValueError('duplicate or unknown prediction ID')
        if type(row['raw_text']) is not str:raise ValueError('raw_text must be a string')
        predicted[key]=row['raw_text']
    output.mkdir(parents=True,exist_ok=False)
    buckets=defaultdict(list);groups=defaultdict(dict);missing=0
    converted=output/'official-input.jsonl'
    with converted.open('w') as handle:
        for row in rows:
            key=row['example_id'];meta=row['metadata']
            missing+=key not in predicted
            pred=predicted.get(key,'MISSING_PREDICTION')
            gold=row['supervision']['official_solution']
            for bucket in ('all',meta['level']):buckets[bucket].append((pred,gold))
            if meta['stage'] in groups[meta['source_group']]:raise ValueError('duplicate conversation stage')
            groups[meta['source_group']][meta['stage']]=strict_metrics([pred],[gold])['exact_correct']
            handle.write(json.dumps({'predict':pred,'label':gold,
                'audio_path':row['metadata']['relative_audios'][-1]},ensure_ascii=False)+'\n')
    metrics={bucket:strict_metrics([p for p,g in pairs],[g for p,g in pairs]) for bucket,pairs in buckets.items()}
    conversation={}
    for level in ('level_3_1','level_3_2'):
        eligible=[stages for group,stages in groups.items() if group.startswith(level+'/')]
        if any(set(stages)!={1,2} for stages in eligible):raise ValueError('incomplete multi-turn group')
        correct=sum(all(stages.values()) for stages in eligible)
        conversation[level]={'count':len(eligible),'both_turns_exact':correct,
                             'accuracy':correct/len(eligible) if eligible else None}
    command=[sys.executable,str(official_scorer),str(converted),'--format','sfc',
             '--output_file',str(output/'official-details.jsonl')]
    with (output/'official-metrics.txt').open('w') as handle:
        subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,check=True)
    report={'manifest_sha256':file_hash(manifest),'predictions_sha256':file_hash(predictions),
            'official_scorer_sha256':expected_scorer_hash,'official_command':command,
            'official_split':next(iter(splits)),'missing_predictions':missing,
            'strict_metrics':metrics,'strict_conversation_metrics':conversation,
            'scope':'offline official supplied history; no claim of interactive tool execution',
            'official_metrics':'official-metrics.txt'}
    atomic_json(output/'metrics.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('manifest','predictions','official-scorer','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-scorer-hash',default='fadf9ffd6a446dbe4e95b5f86b9a8998f5bba88cd515cca50db253962a51bd96')
    a=p.parse_args()
    print(json.dumps(evaluate(a.manifest,a.predictions,a.official_scorer,a.output,a.expected_scorer_hash),indent=2))

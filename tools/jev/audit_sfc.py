#!/usr/bin/env python3
"""Audit official SFC inputs/scorer without model inference or test selection."""
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.sfc import parse_calls, source_identity, strict_metrics


def load_scorer(path):
    spec=importlib.util.spec_from_file_location('official_sfc_metrics',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def official_metrics(scorer,predictions,labels):
    def decode(text):
        raw,error=scorer.robust_parser(text,'sfc')
        return scorer.normalize_sfc(raw)
    return scorer.calculate_metrics([decode(x) for x in predictions],[decode(x) for x in labels])


def audit(root,output):
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    scorer_path=root/'project/eval/metrics_multilevel.py'
    scorer=load_scorer(scorer_path)
    report={'source':'https://github.com/QwenAudio/FunResearch/tree/5cd9dc679584ba9b14454908a0f2c85e74c00147/SpokenFC',
            'version':(root/'VERSION').read_text().strip(),'scorer_sha256':file_hash(scorer_path),
            'scope':'data and scorer audit only; no model prediction or benchmark reproduction',
            'splits':{}}
    hashes={};all_groups={};function_sets={};train_labels=None
    for split in ('train_id','test_id','test_ood'):
        source=root/'data/processed/data_msswift'/f'{split}.json'
        rows=json.loads(source.read_text());groups=defaultdict(list);levels=Counter();functions=set()
        file_ids=set();arrays=0;missing=0;history=0;slots=0;dropped=0
        for row in rows:
            identity=source_identity(row['audios'])
            if identity['official_split']!=split:raise ValueError('split mismatch')
            groups[identity['source_group']].append(identity['stage']);levels[identity['level']]+=1
            if sum(m['content'].count('<audio>') for m in row['messages'])!=len(row['audios']):
                raise ValueError('audio/message count mismatch')
            if row['messages'][-1]['role']!='user':raise ValueError('target in final message')
            history+=len(row['audios'])>1
            for audio in row['audios']:
                if source_identity([audio])['source_group']!=identity['source_group']:
                    raise ValueError('cross-conversation audio')
                path=root/'data'/audio
                if audio not in hashes:hashes[audio]=file_hash(path)
                file_ids.add(audio)
            calls=parse_calls(row['solution'])
            official=scorer.normalize_sfc(scorer.robust_parser(row['solution'],'sfc')[0])
            slots+=sum(len(c['arguments']) for c in calls)
            dropped+=sum(len(c['arguments']) for c in calls)-sum(len(a) for _,a in official)
            for call in calls:
                functions.add(call['name'])
                arrays+=sum(isinstance(v,list) for v in call['arguments'].values())
                missing+=sum(v=='NAN' for v in call['arguments'].values())
        for group,stages in groups.items():
            if sorted(stages)!=([1,2] if group.startswith('level_3') else [1]):
                raise ValueError('duplicate or incomplete conversation')
        all_groups[split]={hashes[a] for a in file_ids};function_sets[split]=functions
        report['splits'][split]={'source_sha256':file_hash(source),'records':len(rows),
            'conversations':len(groups),'levels':dict(levels),'audio_files':len(file_ids),
            'supplied_assistant_history_records':history,'target_functions':len(functions),
            'array_argument_count':arrays,'NAN_argument_count':missing,'strict_argument_count':slots,
            'argument_count_difference_vs_official_parser':dropped}
        if split=='train_id':train_labels=[r['solution'] for r in rows]
    report['audio_hash_overlap']={a+'__'+b:len(all_groups[a]&all_groups[b])
        for a,b in [('train_id','test_id'),('train_id','test_ood'),('test_id','test_ood')]}
    report['target_function_overlap']={a+'__'+b:len(function_sets[a]&function_sets[b])
        for a,b in [('train_id','test_id'),('train_id','test_ood')]}
    report['train_controls']={}
    for name,predictions in [('oracle',train_labels),('empty',['[]']*len(train_labels)),
                             ('malformed',['not JSON']*len(train_labels))]:
        report['train_controls'][name]={'official':official_metrics(scorer,predictions,train_labels),
                                       'strict':strict_metrics(predictions,train_labels)}
    fixtures=[('array_value_changed',['find(x=["Monday"])'],['find(x=["Friday"])']),
              ('slot_binding_swapped',['a(x="one")','b(x="two")'],['a(x="two")','b(x="one")']),
              ('duplicate_call',['a(x="one")','a(x="one")'],['a(x="one")']),
              ('malformed_vs_empty',None,[])]
    report['scorer_diagnostics']={}
    for name,pred,gold in fixtures:
        p='not JSON' if pred is None else json.dumps(pred);g=json.dumps(gold)
        report['scorer_diagnostics'][name]={'official':official_metrics(scorer,[p],[g]),
                                         'strict':strict_metrics([p],[g])}
    atomic_json(output,report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=audit(a.root,a.output)
    print(json.dumps(report,indent=2))

#!/usr/bin/env python3
"""Independently check selection rule truth, provenance and paired inputs."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.slurp_training import content_hash


def verify(root):
    read=lambda p:json.loads(p.read_text())
    manifest=read(root/'MANIFEST.json')
    if (manifest['schema']!='jev-span-selection-rules-v1' or manifest['training_admitted']
            or manifest['evaluation_admitted']):raise ValueError('selection candidate protocol required')
    for name,digest in manifest['files'].items():
        if file_hash(root/name)!=digest:raise ValueError('published selection asset changed')
    for path,digest in manifest['inputs'].items():
        if file_hash(path)!=digest:raise ValueError('pinned review/source changed')
    source=Path(manifest['source']);review_root=Path(manifest['review_root'])
    sm=read(source/'MANIFEST.json')
    for name,digest in sm['files'].items():
        if file_hash(source/name)!=digest:raise ValueError('clean source changed')
    lines=lambda p:[json.loads(s) for s in p.read_text().splitlines()]
    reserved={r['id']:r for r in lines(source/'selection.requests.jsonl')}
    targets={r['id']:r for r in lines(source/'selection.targets.jsonl')}
    learned=lines(source/'learning.requests.jsonl')
    learned_groups={r['source_group'] for r in learned};learned_audio={r['audio_sha256'] for r in learned}
    packet=read(review_root/'packet.json');reviews={r['id']:r for r in read(review_root/'semantic-review.json')['records']}
    tasks_path=next(Path(p) for p in manifest['inputs'] if p.endswith('tasks-v1.json'))
    tasks=read(tasks_path);cases={r['request']['id']:r for r in packet}
    keys=('date','time','timeofday');eligible=set();masks=[]
    for uid,case in cases.items():
        request=reserved[uid];target=targets[uid];review=reviews[uid]
        if (case['request']!=request or case['target']!=target or request['role']!='selection'
                or request['source_group'] in learned_groups or request['audio_sha256'] in learned_audio
                or file_hash(request['input']['audio'])!=request['audio_sha256']
                or review['request_sha256']!=content_hash(request)
                or review['native_target_sha256']!=content_hash(target)
                or review['task_sha256']!={k:content_hash(v) for k,v in tasks.items()}):
            raise ValueError('source/review binding or role isolation differs')
        for i,a in enumerate(keys):
            for b in keys[i+1:]:
                if review['status'][a]==review['status'][b]=='reviewed':eligible.add((uid,a,b))
        for key in keys:
            if review['status'][key]!='reviewed':masks.append((uid,key))
    rows=lines(root/'selection.jsonl');groups=defaultdict(dict);counts=Counter()
    for row in rows:
        p=row['provenance'];uid=p['recording_id'];a,b=p['dependency_keys']
        if (row['split']!='selection' or row['optimization_allowed'] is not False
                or (uid,a,b) not in eligible or p['source_role']!='selection'):
            raise ValueError('nonselection or masked rule admitted')
        review=reviews[uid];request=reserved[uid]
        A,B=(review['reviewed_values'][k] for k in (a,b))
        if any(type(v) is not bool for v in (A,B)) or p['dependency_targets']!=[A,B]:
            raise ValueError('dependency values differ')
        types={e['type'] for e in targets[uid]['native_target']['entities']}
        if A is not (a in types) or B is not (b in types):raise ValueError('unpublished label correction')
        base_id=p['complement_pair_id'];op=base_id.rsplit(':',1)[-1]
        nodes=[dict(op='fact',key=a),dict(op='fact',key=b)]
        if op=='both':value=A and B;rule=dict(op='all',args=nodes)
        elif op=='either':value=A or B;rule=dict(op='any',args=nodes)
        elif op=='first_without_second':value=A and not B;rule=dict(op='except',base=nodes[0],exception=nodes[1])
        else:raise ValueError('unexpected rule operator')
        if base_id!=f'{a}:{b}:{op}':raise ValueError('rule ID does not bind dependencies')
        polarity=p['rule_polarity']
        if polarity=='negative':value=not value;rule=dict(op='not',arg=rule)
        elif polarity!='positive':raise ValueError('unknown polarity')
        if p['rule']!=rule:raise ValueError('rule AST differs')
        expected={'final_only':[value],'structured':[A,B,value],'reordered':[B,A,value]}[row['arm']]
        if row['targets']!=expected or row['supervised']!=[True]*len(expected):raise ValueError('rule gold differs')
        raw=row['request']
        if (set(raw)!= {'audio','questions','final_id','requirements','version'} or raw['audio']!=request['input']['audio']
                or p['source_group']!=request['source_group'] or p['audio_sha256']!=request['audio_sha256']):
            raise ValueError('input provenance or label-free structure differs')
        key=(uid,base_id,polarity)
        if row['arm'] in groups[key]:raise ValueError('duplicate question arm')
        groups[key][row['arm']]=row
        counts[(row['arm'],value)]+=1
    if len(rows)!=len(eligible)*18 or len(rows)!=manifest['records']:raise ValueError('incomplete coverage')
    for (uid,base_id,polarity),arms in groups.items():
        if set(arms)!= {'final_only','structured','reordered'}:raise ValueError('missing question arm')
        a,b,c=(arms[k]['request'] for k in ('final_only','structured','reordered'))
        if (a['questions'][-1]!=b['questions'][-1] or a['questions'][-1]!=c['questions'][-1]
                or a['requirements']!=b['requirements'] or b['requirements']!=c['requirements']
                or [q['schema'] for q in b['questions'][:2]]!=[q['schema'] for q in reversed(c['questions'][:2])]):
            raise ValueError('unmatched question/order controls')
        if polarity=='negative':
            positive=groups[(uid,base_id,'positive')]
            for arm,row in arms.items():
                question=row['request']['questions'][-1]['schema']['question']
                original=positive[arm]['request']['questions'][-1]['schema']['question']
                if not question.startswith('Decide whether the following complete rule is FALSE.') or original not in question:
                    raise ValueError('complement not explicitly visible in model input')
    return dict(root=str(root),passed=True,manifest_sha256=file_hash(root/'MANIFEST.json'),
        recordings=len(cases),rows=len(rows),eligible_recording_fact_pairs=len(eligible),
        masked_decisions=[dict(id=uid,task=key) for uid,key in masks],
        per_arm={arm:dict(false=counts[(arm,False)],true=counts[(arm,True)]) for arm in ('final_only','structured','reordered')},
        source_overlap_with_learning=0,audio_hash_overlap_with_learning=0,
        training_admitted=False,evaluation_admitted=False,
        scope='Independent source/semantic-review bindings and Boolean rule arithmetic; not listening, runtime admission or proof of no historical evaluation exposure')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=verify(a.root);atomic_json(a.output,r);print(json.dumps(r,indent=2))

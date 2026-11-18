#!/usr/bin/env python3
"""Audit source text collisions; equality is not audio or semantic equivalence."""
import argparse
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import unicodedata


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return list(map(json.loads,Path(p).read_text().splitlines()))
def content_sha(v):return hashlib.sha256((json.dumps(v,sort_keys=True,ensure_ascii=False)+'\n').encode()).hexdigest()


def text_key(text,mode):
    s=' '.join(unicodedata.normalize('NFKC',text).casefold().split())
    if mode=='source_normalized':return s
    if mode=='punctuation_relaxed':
        return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c for c in s).split())
    raise ValueError('unregistered normalization')


def summarize(groups,by_source,include_details):
    collisions=[];cross=[];intent=[];slot=[]
    for key,sources in sorted(groups.items()):
        if len(sources)<2:continue
        roles={x['role'] for x in sources}
        entry=dict(text_sha256=hashlib.sha256(key.encode()).hexdigest(),
            source_ids=sorted(x['source_id'] for x in sources),roles=sorted(roles))
        if len(roles)>1:cross.append(entry)
        learn=[x for x in sources if x['role']=='learning']
        if len(learn)<2:continue
        labels={x['intent'] for x in learn};types={tuple(x['types']) for x in learn}
        record_ids={r['id'] for x in learn for r in by_source[x['source_id']]}
        detail=dict(entry,learning_source_ids=sorted(x['source_id'] for x in learn),
            learning_intents=sorted(labels),learning_type_sets=[list(t) for t in sorted(types)],
            recording_ids=sorted(record_ids),source_groups=sorted({r['source_group'] for x in learn for r in by_source[x['source_id']]}))
        if include_details:detail['learning_reference']=key
        collisions.append(detail)
        if len(labels)>1:intent.append(detail)
        if len(types)>1:slot.append(detail)
    union=lambda entries,field:len({v for e in entries for v in e[field]})
    return dict(learning_collision_groups=len(collisions),learning_collision_sources=union(collisions,'learning_source_ids'),
        learning_collision_recordings=union(collisions,'recording_ids'),
        cross_role_collision_groups=len(cross),cross_role_collision_identities=cross,
        learning_intent_disagreement_groups=len(intent),learning_intent_disagreement_sources=union(intent,'learning_source_ids'),
        learning_intent_disagreement_recordings=union(intent,'recording_ids'),
        learning_type_set_disagreement_groups=len(slot),learning_type_set_disagreement_sources=union(slot,'learning_source_ids'),
        learning_type_set_disagreement_recordings=union(slot,'recording_ids'),
        intent_disagreements=intent,type_set_disagreements=slot)


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if sha(a.release/'MANIFEST.json')!=a.manifest_sha256:raise ValueError('release identity mismatch')
    m=json.loads((a.release/'MANIFEST.json').read_text());source=Path(m['source']);sm=json.loads((source/'MANIFEST.json').read_text())
    mp=source.parent/'metadata/train.jsonl'
    pins={str(a.release/'MANIFEST.json'):a.manifest_sha256,str(source/'MANIFEST.json'):sha(source/'MANIFEST.json'),str(mp):sm['identity']['input_hashes']['clean_metadata_sha256']}
    for role in ('learning','selection'):
        pins[str(a.release/(role+'.jsonl'))]=m['files'][role+'.jsonl']
        pins[str(source/(role+'.targets.jsonl'))]=sm['files'][role+'.targets.jsonl']
    for path,h in pins.items():
        if sha(path)!=h:raise ValueError('changed dependency '+path)
    by_source=defaultdict(list);source_role={};counts={}
    for role in ('learning','selection'):
        published=rows(a.release/(role+'.jsonl'));counts[role]=len(published)
        if len({r['id'] for r in published})!=len(published):raise ValueError('duplicate recording')
        targets={r['id']:r for r in rows(source/(role+'.targets.jsonl'))}
        for r in published:
            if r['role']!=role or r['official_split']!='train' or r['known_pending_source']:raise ValueError('invalid population')
            t=targets[r['id']]
            if content_sha(t)!=r['source_target_sha256']:raise ValueError('target mismatch')
            for sid in t['source_ids']:
                sid=str(sid)
                if sid in source_role and source_role[sid]!=role:raise ValueError('cross role source ID')
                source_role[sid]=role;by_source[sid].append(r)
    if counts!={'learning':40513,'selection':9226}:raise ValueError('incomplete population')
    groups={mode:defaultdict(list) for mode in ('source_normalized','punctuation_relaxed')}
    for row in rows(mp):
        sid=str(row['slurp_id'])
        if sid not in source_role:continue
        role=source_role[sid]
        # Selection references used ONLY for identity comparison, never rendered or annotated.
        item=dict(source_id=sid,role=role)
        if role=='learning':item.update(intent=row['scenario']+'_'+row['action'],types=sorted({e['type'] for e in row['entities']}))
        for mode in groups:groups[mode][text_key(row['sentence'],mode)].append(item)
    result={mode:summarize(g,by_source,True) for mode,g in groups.items()}
    out=dict(schema='slurp-text-collision-audit-v1',pins=pins,implementation_sha256=sha(__file__),recordings=counts,
        source_counts=dict(Counter(source_role.values())),normalizations=result,training_dispatch_allowed=False,
        scope='Learning text/annotation triage; selection text used only for overlap identities, no selection label analysis or definition authoring. Normalized text equality is not audio equivalence; punctuation-relaxed collisions are diagnostic, not automatic exclusions.')
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    for mode,r in result.items():
        print(mode,json.dumps({k:v for k,v in r.items() if not isinstance(v,list)}))
        for x in r['intent_disagreements']:print('INTENT',x['learning_reference'],x['learning_source_ids'],x['learning_intents'])
        for x in r['type_set_disagreements']:print('TYPES',x['learning_reference'],x['learning_source_ids'],x['learning_type_sets'])


if __name__=='__main__':main()

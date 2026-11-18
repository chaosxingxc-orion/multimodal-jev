#!/usr/bin/env python3
"""Expand selected native-intent learning cohorts for complete reference review."""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def content_sha(v):
    return hashlib.sha256((json.dumps(v,sort_keys=True,ensure_ascii=False)+'\n').encode()).hexdigest()


def rows(p):
    return list(map(json.loads,Path(p).read_text().splitlines()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evidence-packet',type=Path,required=True)
    p.add_argument('--packet-sha256',required=True)
    p.add_argument('--intents',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if sha(a.evidence_packet)!=a.packet_sha256:raise ValueError('packet identity differs')
    packet=json.loads(a.evidence_packet.read_text());pins=packet['pins']
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('changed source: '+path)
    find=lambda suffix:next(Path(k) for k in pins if k.endswith(suffix))
    learning=rows(find('/learning.jsonl'))
    targets={r['id']:r for r in rows(find('/learning.targets.jsonl'))}
    chosen=set(a.intents)
    if not chosen<={c['semantic_id'] for c in packet['cards']}:raise ValueError('unknown class')
    selected=[r for r in learning if r['intent'] in chosen]
    by_source=defaultdict(list)
    for r in selected:
        if r['role']!='learning' or r['official_split']!='train' or r['known_pending_source']:raise ValueError('invalid role')
        t=targets[r['id']]
        if content_sha(t)!=r['source_target_sha256']:raise ValueError('target mismatch')
        for sid in t['source_ids']:by_source[str(sid)].append(r)
    meta={str(r['slurp_id']):r for r in rows(find('/metadata/train.jsonl')) if str(r['slurp_id']) in by_source}
    if set(meta)!=set(by_source):raise ValueError('missing source')
    result=[]
    for sid in sorted(meta,key=int):
        r=meta[sid];label=r['scenario']+'_'+r['action']
        if any(x['intent']!=label for x in by_source[sid]):raise ValueError('label disagreement')
        result.append(dict(source_id=sid,native_intent=label,reference=r['sentence'],
            metadata_sha256=content_sha(r),recordings=[dict(id=x['id'],source_group=x['source_group'],
                audio_sha256=x['audio_sha256'],target_sha256=x['source_target_sha256'],
                all_source_ids=targets[x['id']]['source_ids']) for x in by_source[sid]]))
    out=dict(schema='slurp-complete-intent-cohort-evidence-v1',pins=pins,
        packet=str(a.evidence_packet),packet_sha256=a.packet_sha256,implementation_sha256=sha(__file__),
        native_intents=sorted(chosen),source_utterances=len(result),recordings=len(selected),
        counts=dict(Counter(x['intent'] for x in selected)),sources=result,
        scope='Complete selected native-class learning cohorts; not all semantic cooking queries across all classes; no audio listening or supervision admission.',training_dispatch_allowed=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    for x in result:print(x['source_id'],x['native_intent'],len(x['recordings']),x['reference'],sep='\t')


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Check learning-only raw intent versus canonical scenario/action identities."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):return list(map(json.loads,Path(p).read_text().splitlines()))
def content_sha(v):return hashlib.sha256((json.dumps(v,sort_keys=True,ensure_ascii=False)+'\n').encode()).hexdigest()


def resolve(record,pairs):
    pair=(record['scenario'],record['action'])
    reverse={tuple(v):k for k,v in pairs.items()}
    if len(reverse)!=len(pairs):raise ValueError('non-injective candidate mapping')
    if pair not in reverse:raise ValueError('pair outside frozen ontology')
    return reverse[pair]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--original-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if sha(a.release/'MANIFEST.json')!=a.manifest_sha256 or sha(a.original)!=a.original_sha256:raise ValueError('source pin mismatch')
    m=read(a.release/'MANIFEST.json');source=Path(m['source']);sm=read(source/'MANIFEST.json');mp=source.parent/'metadata/train.jsonl'
    pins={str(a.release/'MANIFEST.json'):a.manifest_sha256,str(a.original):a.original_sha256,
          str(source/'MANIFEST.json'):sha(source/'MANIFEST.json'),str(mp):sm['identity']['input_hashes']['clean_metadata_sha256'],
          str(source/'learning.targets.jsonl'):sm['files']['learning.targets.jsonl']}
    for name in ('learning.jsonl','tasks.json'):pins[str(a.release/name)]=m['files'][name]
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('changed source '+path)
    learning=rows(a.release/'learning.jsonl');targets={r['id']:r for r in rows(source/'learning.targets.jsonl')}
    if len(learning)!=40513 or len({r['id'] for r in learning})!=40513:raise ValueError('complete population required')
    by_source=defaultdict(list);pairs=read(a.release/'tasks.json')['intent']['candidate_pairs']
    for r in learning:
        if r['role']!='learning' or r['official_split']!='train' or r['known_pending_source']:raise ValueError('learning only')
        t=targets[r['id']]
        if content_sha(t)!=r['source_target_sha256'] or resolve(t['native_target'],pairs)!=r['intent']:raise ValueError('target mismatch')
        for sid in t['source_ids']:by_source[str(sid)].append(r)
    meta={str(r['slurp_id']):r for r in rows(mp) if str(r['slurp_id']) in by_source}
    original={str(r['slurp_id']):r for r in rows(a.original) if str(r['slurp_id']) in by_source}
    if set(meta)!=set(by_source) or set(original)!=set(by_source):raise ValueError('missing source')
    cross=Counter();differences=[];raw_to=defaultdict(set);changed=[];suffix_fail=[]
    for sid,r in sorted(meta.items(),key=lambda kv:int(kv[0])):
        canonical=resolve(r,pairs);raw=r['intent'];cross[(raw,canonical)]+=1;raw_to[raw].add(canonical)
        if any(x['intent']!=canonical for x in by_source[sid]):raise ValueError('source/recording canonical mismatch')
        if original[sid]!=r:changed.append(sid)
        if raw!=canonical:
            action_alias=raw==r['action']
            if not action_alias:suffix_fail.append(sid)
            differences.append(dict(source_id=sid,raw_intent=raw,scenario=r['scenario'],action=r['action'],
                canonical_intent=canonical,raw_equals_action=action_alias,metadata_sha256=content_sha(r),
                original_equals_clean=original[sid]==r,recording_ids=[x['id'] for x in by_source[sid]]))
    if suffix_fail:raise ValueError('unexpected raw namespace discrepancy: '+str(suffix_fail))
    report=dict(schema='slurp-learning-intent-namespace-audit-v1',pins=pins,implementation_sha256=sha(__file__),
        recordings=len(learning),source_utterances=len(meta),raw_intent_strings=len(raw_to),canonical_classes=len(pairs),
        mismatch_source_count=len(differences),mismatch_recordings=len({x for d in differences for x in d['recording_ids']}),
        raw_alias_strings=len({x['raw_intent'] for x in differences}),
        all_mismatches_are_action_only_aliases=not suffix_fail,
        ambiguous_raw_aliases={k:sorted(v) for k,v in sorted(raw_to.items()) if len(v)>1},
        original_vs_clean_changed_learning_sources=changed,
        crosswalk=[dict(raw_intent=a,canonical_intent=b,source_count=n) for (a,b),n in sorted(cross.items())],
        differences=differences,training_dispatch_allowed=False,
        conclusion='Canonical scenario/action mapping is authenticated for the full learning population. Raw intent alone is not a valid key. Namespace repair does not resolve semantic definition conflicts or admit new Q labels.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('pins','crosswalk','differences')},indent=2))


if __name__=='__main__':main()

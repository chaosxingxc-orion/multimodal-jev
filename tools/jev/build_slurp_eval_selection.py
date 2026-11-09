#!/usr/bin/env python3
"""Freeze SLURP dev diagnostics before generation; preserve every selected view."""
import argparse
from collections import defaultdict,Counter
import hashlib,json,os,tempfile,shutil
from pathlib import Path
from multimodal_jev.baseline import file_hash
from multimodal_jev.task_schema import JevExample


def select(rows):
    groups=defaultdict(list)
    for row in rows:
        ex=JevExample.from_record(row)
        if ex.split!='valid':raise ValueError('official development split only')
        groups[ex.metadata['source_group']].append(row)
    expected={'native','permutation','rebind','slot.date','slot.time','slot.person','slot.place_name'}
    buckets=defaultdict(set)
    for group,views in groups.items():
        if len(views)!=7 or {v['metadata']['variant'] for v in views}!=expected:
            raise ValueError('complete seven-view groups required')
        meta=views[0]['metadata']
        buckets['scenario:'+meta['scenario']].add(group)
        for row in views:
            variant=row['metadata']['variant']
            if variant.startswith('slot.'):
                buckets[meta['partition']+':'+variant+(':.null' if row['target'] is None else ':.present')].add(group)
    # One group per scenario, plus one per partition/slot/presence where available.
    # Selection uses hashes, not scores, duration or model output.
    chosen={min(values,key=lambda g:hashlib.sha256(g.encode()).digest()) for values in buckets.values()}
    selected=[row for group in sorted(chosen) for row in groups[group]]
    support=Counter((r['metadata']['partition'],r['metadata']['variant'],r['target'] is not None) for r in selected)
    return selected,{key:min(values,key=lambda g:hashlib.sha256(g.encode()).digest()) for key,values in sorted(buckets.items())},support


def build(source,fsc_train,destination):
    rows=[]
    for partition in ('valid','heldout-valid'):
        with (source/(partition+'.jsonl')).open() as f:rows.extend(json.loads(line) for line in f)
    selected,buckets,support=select(rows)
    fsc_audio=set();fsc_groups=set()
    with fsc_train.open() as f:
        for line in f:
            meta=json.loads(line)['metadata'];fsc_audio.add(meta['audio_sha256']);fsc_groups.add(meta['source_group'])
    if fsc_audio&{r['metadata']['audio_sha256'] for r in selected}:raise ValueError('FSC audio overlap')
    if fsc_groups&{r['metadata']['source_group'] for r in selected}:raise ValueError('FSC source overlap')
    payload=''.join(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n' for row in selected).encode()
    release=hashlib.sha256(payload).hexdigest();final=destination/release
    if final.exists():raise FileExistsError(final)
    destination.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.selection-',dir=destination))
    try:
        (stage/'valid.jsonl').write_bytes(payload)
        receipt={'release_id':release,'source_sha256':{p:file_hash(source/(p+'.jsonl')) for p in ('valid','heldout-valid')},
                 'fsc_train_sha256':file_hash(fsc_train),'groups':len({r['metadata']['source_group'] for r in selected}),
                 'utterances':len({r['metadata']['utterance_group'] for r in selected}),
                 'speakers':len({r['metadata']['speaker_id'] for r in selected}),'decisions':len(selected),
                 'selection_buckets':buckets,'support':{'/'.join((p,v,'present' if yes else 'null')):n for (p,v,yes),n in sorted(support.items())},
                 'fsc_audio_overlap':0,'builder_sha256':file_hash(__file__),
                 'scope':'stratified development diagnostic, deliberately enriched for non-null slots; not prevalence estimate or final test'}
        (stage/'MANIFEST.json').write_text(json.dumps(receipt,indent=2)+'\n')
        stage.chmod(0o755);os.rename(stage,final)
    except BaseException:shutil.rmtree(stage);raise
    return final,receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--fsc-train',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args();path,receipt=build(a.source,a.fsc_train,a.output_root)
    print(json.dumps({'path':str(path),**receipt},indent=2))

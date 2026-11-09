#!/usr/bin/env python3
"""Freeze one official valid source group per native class and all six views."""
import argparse,json,hashlib,os,tempfile,shutil
from pathlib import Path
from multimodal_jev.baseline import select_examples,file_hash


def build(manifest,training_identity,destination):
    train=json.loads(training_identity.read_text())
    native=select_examples(manifest,'valid',1)
    groups={ex.metadata['source_group'] for ex in native}
    rows=[]
    with manifest.open() as handle:
        for line in handle:
            row=json.loads(line)
            if row['metadata']['source_group'] in groups:rows.append(row)
    if len(rows)!=len(groups)*6:raise ValueError('six views per group required')
    train_ids=set(train['example_ids'])
    if train_ids&{r['example_id'] for r in rows}:raise ValueError('training/evaluation ID overlap')
    train_groups={x.rsplit(':',1)[0] for x in train_ids}
    if train_groups&groups:raise ValueError('training/evaluation source-group overlap')
    payload=''.join(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n' for row in rows).encode()
    release=hashlib.sha256(payload).hexdigest()
    root=destination/release
    if root.exists():raise FileExistsError('selection already exists')
    destination.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.selection-',dir=destination))
    try:
        (stage/'valid.jsonl').write_bytes(payload)
        receipt={'release_id':release,'source_manifest':str(manifest),'source_sha256':file_hash(manifest),
                 'training_identity_sha256':file_hash(training_identity),'groups':len(groups),'decisions':len(rows),
                 'selection':'minimum sha256(native example_id) per train-defined label on official valid',
                 'builder_sha256':file_hash(__file__),'source_group_overlap_with_sft':0}
        (stage/'MANIFEST.json').write_text(json.dumps(receipt,indent=2)+'\n')
        os.rename(stage,root)
    except BaseException:
        shutil.rmtree(stage);raise
    return root,receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--training-identity',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args();root,receipt=build(a.manifest,a.training_identity,a.output_root)
    print(json.dumps({'path':str(root),**receipt},indent=2))

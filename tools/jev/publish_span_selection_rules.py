#!/usr/bin/env python3
"""Publish source-disjoint selection controls, never a training release."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_selection_review import select_review_packet
from multimodal_jev.span_selection_rules import project_selection


def publish(review_root,tasks_path,output):
    read=lambda p:json.loads(p.read_text())
    receipt=read(review_root/'receipt.json');source=Path(receipt['source'])
    manifest=read(source/'MANIFEST.json')
    if file_hash(source/'MANIFEST.json')!=receipt['source_manifest_sha256']:raise ValueError('source changed')
    for name,digest in manifest['files'].items():
        if file_hash(source/name)!=digest:raise ValueError('source data changed')
    metadata=Path(manifest['source_root'])/'metadata/train.jsonl'
    if file_hash(metadata)!=receipt['metadata_sha256']:raise ValueError('metadata changed')
    lines=lambda p:[json.loads(s) for s in p.read_text().splitlines()]
    packet=read(review_root/'packet.json');review=read(review_root/'semantic-review.json');tasks=read(tasks_path)
    if (file_hash(review_root/'packet.json')!=receipt['packet_sha256']
            or review['packet_sha256']!=receipt['packet_sha256']
            or review['tasks_sha256']!=file_hash(tasks_path)):
        raise ValueError('review/task binding changed')
    recreated,_=select_review_packet(lines(source/'selection.requests.jsonl'),lines(source/'selection.targets.jsonl'),
        lines(metadata),lines(source/'learning.requests.jsonl'),receipt['requested_per_native_stratum'],receipt['seed'])
    if packet!=recreated:raise ValueError('selection population differs from fixed sampling')
    for row in packet:
        if file_hash(row['request']['input']['audio'])!=row['request']['audio_sha256']:raise ValueError('audio changed')
    rows,audit=project_selection(packet,review,tasks)
    output.mkdir(parents=True,exist_ok=False)
    (output/'selection.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    atomic_json(output/'audit.json',audit)
    # Separate review evidence from actor inputs and task targets.
    inputs=[review_root/n for n in ('packet.json','receipt.json','semantic-review.json')]+[tasks_path,source/'MANIFEST.json']
    report=dict(schema='jev-span-selection-rules-v1',records=len(rows),recordings=len(packet),
        review_scope=review['scope'],training_admitted=False,evaluation_admitted=False,
        source_overlap_with_learning=0,audio_hash_overlap_with_learning=0,
        sampling='stratified source selection; rare strata retained without replacement',
        test_opened=False,source=str(source),review_root=str(review_root),
        inputs={str(p.resolve()):file_hash(p) for p in inputs},
        files={p.name:file_hash(p) for p in output.iterdir()},
        scope='Selection diagnostics candidate release; not official-test distribution or automatically admitted model evaluation')
    atomic_json(output/'MANIFEST.json',report)
    for p in (output,*output.iterdir()):p.chmod(0o777 if p.is_dir() else 0o644)
    return dict(root=str(output),manifest_sha256=file_hash(output/'MANIFEST.json'),**report)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('review-root','tasks','output','receipt'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();report=publish(a.review_root,a.tasks,a.output);atomic_json(a.receipt,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','files')},indent=2))

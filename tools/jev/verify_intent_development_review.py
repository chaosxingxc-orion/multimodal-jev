#!/usr/bin/env python3
"""Validate a complete offline screening ledger without granting admission."""
import argparse
from collections import Counter
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.slurp_training import content_hash


def verify(packet_path,audit_path):
    packet_path,audit_path=map(Path,(packet_path,audit_path))
    packet=json.loads(packet_path.read_text());audit=json.loads(audit_path.read_text())
    if audit['packet_sha256']!=file_hash(packet_path):raise ValueError('review packet changed')
    if len(audit['records'])!=len(packet):raise ValueError('every original record requires a decision')
    counts=Counter();classes=Counter();pending_classes=Counter()
    for i,(source,row) in enumerate(zip(packet,audit['records'],strict=True)):
        target=source['target']['native_target'];label=target['scenario']+'_'+target['action']
        if (row['index']!=i or row['id']!=source['request']['id']
                or row['source_group']!=source['request']['source_group'] or row['intent']!=label
                or row['record_sha256']!=content_hash(source)):
            raise ValueError('review source identity differs')
        if (row['status'] not in ('reference_consistent','pending') or not row['reason'].strip()
                or any(row[k] is not False for k in ('audio_reviewed','evaluation_allowed','optimization_allowed'))):
            raise ValueError('reference screening cannot grant audio/evaluation/training admission')
        counts[row['status']]+=1
        (classes if row['status']=='reference_consistent' else pending_classes)[label]+=1
    if any(audit[k]!=counts[k] for k in ('reference_consistent','pending')):raise ValueError('review counts differ')
    return dict(passed=True,recordings=len(packet),counts=dict(counts),
        reference_consistent_classes=len(classes),reference_consistent_by_class=dict(sorted(classes.items())),
        pending_by_class=dict(sorted(pending_classes.items())),
        evaluation_allowed=False,optimization_allowed=False,
        files={str(p.resolve()):file_hash(p) for p in (packet_path,audit_path)},
        scope='Complete reference screening ledger integrity only; no audio adjudication or benchmark admission.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--packet',required=True);p.add_argument('--audit',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable receipt exists')
    atomic_json(a.output,verify(a.packet,a.audit))

#!/usr/bin/env python3
"""Validate listening-ledger identity/completeness; never certify semantics."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.slurp_training import content_hash
from verify_intent_development_review import verify as verify_reference


def validate(packet,ledger,packet_hash,reference_hash):
    if (ledger.get('schema')!='slurp-intent-listening-review-v1' or ledger.get('packet_sha256')!=packet_hash
            or ledger.get('reference_audit_sha256')!=reference_hash
            or ledger.get('evaluation_allowed') is not False or ledger.get('optimization_allowed') is not False):
        raise ValueError('exact unapproved listening ledger required')
    rows=ledger['records'];by_id={r['id']:r for r in rows}
    if len(rows)!=len(packet) or len(by_id)!=len(rows) or set(by_id)!={s['request']['id'] for s in packet}:
        raise ValueError('complete distinct original population required')
    counts=Counter();reviewers=set();completed=0
    for source in packet:
        r=source['request'];row=by_id[r['id']]
        if (row['source_group']!=r['source_group'] or row['audio_sha256']!=r['audio_sha256']
                or row['record_sha256']!=content_hash(source)):
            raise ValueError('listening record identity differs')
        if row.get('evaluation_allowed') is not False or row.get('optimization_allowed') is not False:
            raise ValueError('listening ledger cannot grant admission')
        alignment=row['alignment'];ontology=row['ontology']
        if alignment not in ('pending','matches','mismatch','unintelligible') or ontology not in ('pending','consistent','conflict'):
            raise ValueError('explicit listening states required')
        active=alignment!='pending' or ontology!='pending'
        if active:
            if not row.get('notes','').strip() or not row.get('reviewer','').strip():
                raise ValueError('review attribution and reason required')
            try:stamp=datetime.fromisoformat(row['recorded_at'].replace('Z','+00:00'))
            except (KeyError,ValueError,TypeError):raise ValueError('valid listening timestamp required')
            if stamp.tzinfo is None:raise ValueError('timezone required')
            reviewers.add(row['reviewer']);completed+=alignment!='pending' and ontology!='pending'
        counts[alignment+'/'+ontology]+=1
    return dict(passed=True,recordings=len(packet),both_fields_reviewed=completed,reviewers=sorted(reviewers),states=dict(sorted(counts.items())),
        evaluation_allowed=False,optimization_allowed=False,scope='Ledger structural integrity only. Review assertions are not independent semantic evidence or benchmark admission.')


def run(packet_path,reference_path,listening_path):
    packet_path,reference_path,listening_path=map(Path,(packet_path,reference_path,listening_path))
    verify_reference(packet_path,reference_path)
    read=lambda p:json.loads(p.read_text())
    result=validate(read(packet_path),read(listening_path),file_hash(packet_path),file_hash(reference_path))
    result['files']={str(p.resolve()):file_hash(p) for p in (packet_path,reference_path,listening_path)}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('packet','reference','listening','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('immutable review verification exists')
    atomic_json(out,run(a.packet,a.reference,a.listening))

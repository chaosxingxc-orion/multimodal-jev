#!/usr/bin/env python3
"""Prepare bounded local feedback after complete probe acceptance; never call API."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_teacher_feedback import TEMPLATE,build_feedback
from verify_sfc_jev_probe import accept


def prepare(root,partition_path,expected_partition_hash,output):
    if file_hash(partition_path)!=expected_partition_hash:raise ValueError('registered partition changed')
    partition=json.loads(partition_path.read_text())
    if json.loads((root/'evaluation-terminal.json').read_text())['exit_code']!=0:
        raise ValueError('completed successful Jev probe required')
    rows,preflight,raw=accept(root)
    accepted=json.loads((root/'acceptance.json').read_text())
    if accepted['metrics_sha256']!=file_hash(root/'score/metrics.json'):
        raise ValueError('probe scoring acceptance changed')
    release=Path(preflight['release']);manifest=json.loads((release/'MANIFEST.json').read_text())
    source=release/'train_id.requests.jsonl';target=release/'train_id.targets.jsonl'
    if (partition['release']!=str(release.resolve()) or partition['manifest_sha256']!=file_hash(release/'MANIFEST.json')
            or partition['train_requests_sha256']!=file_hash(source)
            or file_hash(source)!=manifest['splits']['train_id']['requests_sha256']
            or file_hash(target)!=manifest['splits']['train_id']['targets_sha256']):
        raise ValueError('partition/source/target custody changed')
    all_rows=[json.loads(line) for line in source.open()];ids={r['id'] for r in rows}
    targets=[t for t in map(json.loads,target.open()) if t['id'] in ids]
    predictions=[{k:p[k] for k in ('example_id','decision_id','request_sha256','raw_text')} for p in raw]
    feedback=build_feedback(all_rows,partition,rows,targets,predictions)
    data=canonical(feedback)
    if len(data)>150000:raise ValueError('feedback budget exceeded; do not silently truncate')
    output.mkdir(parents=True,exist_ok=False,mode=0o700)
    messages=output/'messages.private.json'
    messages.write_bytes(canonical([dict(role='system',content=TEMPLATE),dict(role='user',content=data.decode())]));messages.chmod(0o600)
    receipt=dict(API_sent=False,partition_sha256=file_hash(partition_path),probe_result_sha256=file_hash(root/'run/result.json'),
        probe_score_sha256=file_hash(root/'score/metrics.json'),messages_sha256=file_hash(messages),context_bytes=len(data),
        bundles=len(rows),decisions=len(raw),representatives=len(feedback['representatives']),
        selection_or_test_feedback_used=False,max_candidates=2,max_instruction_characters=1500,
        scope='Local preparation only. No API budget or executor dispatch. Generic prompt adaptation still requires '
              'candidate integration, engineering verification and independent selection.')
    atomic_json(output/'prepared.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','partition','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--expected-partition-hash',required=True);a=p.parse_args()
    print(json.dumps(prepare(a.root,a.partition,a.expected_partition_hash,a.output),indent=2))

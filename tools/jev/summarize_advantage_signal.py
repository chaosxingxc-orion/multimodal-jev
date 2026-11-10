#!/usr/bin/env python3
"""Summarize a completed pinned-verl advantage audit and its saved rollout."""
import argparse
import json
from pathlib import Path
from multimodal_jev.advantage_summary import summarize_signal
from multimodal_jev.baseline import file_hash
from multimodal_jev.task_schema import JevExample


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rollout',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('existing diagnosis retained')
    for run in (a.rollout,a.audit):
        if json.loads((run/'terminal.json').read_text())['exit_code']!=0:
            raise ValueError('successful rollout and audit terminals required')
    identity=json.loads((a.rollout/'identity.json').read_text())
    audit=json.loads((a.audit/'result.json').read_text())
    config=identity['config'];wanted=set(config['example_ids'])
    if (audit['rollout_sha256']!=file_hash(a.rollout/'rollouts.jsonl') or
            audit['policy_id']!=identity['policy_id'] or
            audit['replicates']!=config['sampling']['n'] or
            file_hash(config['manifest'])!=identity['manifest_sha256']):
        raise ValueError('audit/rollout identity changed')
    records=[json.loads(line) for line in (a.rollout/'rollouts.jsonl').open()]
    if [(r['example_id'],r['replicate']) for r in records]!=[
            (r['example_id'],r['replicate']) for r in audit['per_response']]:
        raise ValueError('advantage audit response order differs from rollout')
    examples=[]
    with Path(config['manifest']).open() as handle:
        for line in handle:
            row=json.loads(line)
            if row['example_id'] in wanted:examples.append(JevExample.from_record(row))
    if {ex.example_id for ex in examples}!=wanted:
        raise ValueError('manifest selection missing')
    report=summarize_signal(audit,examples)
    report.update(audit_sha256=file_hash(a.audit/'result.json'),
                  rollout_sha256=audit['rollout_sha256'],policy_id=audit['policy_id'])
    with a.output.open('x') as handle:json.dump(report,handle,indent=2,allow_nan=False)


if __name__=='__main__':main()

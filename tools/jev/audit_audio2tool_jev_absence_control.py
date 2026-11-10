#!/usr/bin/env python3
"""Analytic always-absent control, not a model or a training-weight estimate."""
import argparse
import json
from pathlib import Path

from multimodal_jev.audio2tool_inference import write_once
from multimodal_jev.baseline import file_hash

MANIFEST_SHA='56921792b06636d83bd6b8a21e49ae0f887f3de68a9813d75fe1b98d83c0f062'


def counts(target,tool_names):
    present={target['primary']['name'],*(a['name'] for a in target['additional_annotations'])}
    missing={a['name'] for a in target['additional_annotations'] if a['argument_evidence']=='unavailable'}
    if not present or not present<=tool_names or not missing<=present:
        raise ValueError('known nonempty tool names and explicit missing arguments required')
    return dict(required_tools=len(present),absent_tools=len(tool_names-present),
        unavailable_call_lists=len(missing),available_required_call_lists=len(present-missing))


def run(args):
    if file_hash(args.release/'MANIFEST.json')!=MANIFEST_SHA:raise ValueError('fixed complete Jev release required')
    manifest=json.loads((args.release/'MANIFEST.json').read_text())
    if (file_hash(args.release/'external_test.targets.jsonl')!=manifest['targets_sha256']
            or file_hash(args.release/'task-bank.json')!=manifest['task_bank_file_sha256']):
        raise ValueError('unaltered full target and registry bank required')
    bank=json.loads((args.release/'task-bank.json').read_text());names={t['tool_name'] for t in bank['tools']}
    totals=dict(required_tools=0,absent_tools=0,unavailable_call_lists=0,available_required_call_lists=0)
    n=0
    for line in (args.release/'external_test.targets.jsonl').open():
        row=json.loads(line);n+=1
        for k,v in counts(row['native_target'],names).items():totals[k]+=v
    if n!=30733 or len(names)!=152 or totals['unavailable_call_lists']!=6623:
        raise ValueError('complete registered decision denominators required')
    present,absent,unknown=(totals[k] for k in ('required_tools','absent_tools','unavailable_call_lists'))
    return dict(status='analytic_control_complete',records=n,tool_decisions=present+absent,
        control='For every tool: present=false, absent=true, calls=[]; tools.sequence=[].',counts=totals,
        presence_confusion=dict(tp=0,fp=0,fn=present,tn=absent),
        semantic_presence_recall=0.,semantic_presence_f1=0.,presence_balanced_accuracy=.5,
        pooled_noul_accuracy=absent/(present+absent),
        typed_call_list_accuracy_on_available=absent/(present+absent-unknown),
        typed_call_list_available_denominator=present+absent-unknown,
        tool_sequence_exact=0.,full_population_primary_tool_accuracy=0.,
        full_argument_subset_exact=dict(records=25969,value=0.),
        manifest_sha256=MANIFEST_SHA,runner_sha256=file_hash(__file__),model_loaded=False,
        scope='Closed-form always-absent degeneration on all gold records, with absent/present polarity treated semantically. No synthetic model prediction files, training weights, prompt selection, actor feedback or measured model performance.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('preserve earlier control')
    write_once(args.output,run(args))

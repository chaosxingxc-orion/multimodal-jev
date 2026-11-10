#!/usr/bin/env python3
"""Independent complete-roster audit of compact Audio2Tool Jev artifacts."""
import argparse
from collections import Counter
import json
from pathlib import Path

from multimodal_jev.audio2tool_acceptance import TARGETS_SHA
from multimodal_jev.audio2tool_input import REGISTRY_SHA256
from multimodal_jev.audio2tool_inference import INPUT_SHA,digest,write_once
from multimodal_jev.audio2tool_jev import task_bank,PROTOCOL
from multimodal_jev.audio2tool_score import typed
from multimodal_jev.baseline import file_hash


def verify(args):
    manifest=json.loads((args.release/'MANIFEST.json').read_text());identity=manifest['identity']
    if (file_hash(args.inputs)!=INPUT_SHA or file_hash(args.targets)!=TARGETS_SHA
            or file_hash(args.registry)!=REGISTRY_SHA256 or identity['inputs_sha256']!=INPUT_SHA
            or identity['targets_sha256']!=TARGETS_SHA or identity['registry_sha256']!=REGISTRY_SHA256
            or identity['split']!='external_test' or digest(identity)!=manifest['release_id']
            or args.release.name!=manifest['release_id'] or json.loads((args.release/'terminal.json').read_text())['exit_code']!=0):
        raise ValueError('registered source and complete content-addressed release required')
    bank=json.loads((args.release/'task-bank.json').read_text())
    if bank!=task_bank(args.registry.read_bytes()) or digest(bank)!=identity['task_bank_sha256']:
        raise ValueError('input-only independent task bank differs')
    for name,key in [('task-bank.json','task_bank_file_sha256'),('external_test.requests.jsonl','requests_sha256'),
                     ('external_test.targets.jsonl','targets_sha256')]:
        if file_hash(args.release/name)!=manifest[key]:raise ValueError('release artifact changed')
    tools={t['tool_name']:t['tool_id'] for t in bank['tools']};counts=Counter();unavailable=0;complete=0;ids=set();unique=set()
    with args.inputs.open() as ni,args.targets.open() as nt, \
            (args.release/'external_test.requests.jsonl').open() as ji, \
            (args.release/'external_test.targets.jsonl').open() as jt:
        for nline,tline,rline,gline in zip(ni,nt,ji,jt,strict=True):
            native,target,row,gold=map(json.loads,(nline,tline,rline,gline))
            expected=dict(id=native['id'],tier=native['tier'],split='external_test',protocol=PROTOCOL,
                native_request_sha256=digest(native),input=dict(context=native['input'],task_bank_sha256=identity['task_bank_sha256']))
            if (row!=expected or native['split']!='external_test' or target['id']!=row['id']
                    or set(gold)!={'id','request_sha256','native_target','unavailable_decisions'}
                    or gold['id']!=row['id'] or gold['request_sha256']!=digest(row)
                    or typed(gold['native_target'])!=typed(target) or row['id'] in ids):
                raise ValueError('unaltered full source row, compact target and actor separation required')
            missing_names={a['name'] for a in target['additional_annotations'] if a['argument_evidence']=='unavailable'}
            missing=['tool.'+t['tool_id']+'.calls' for t in bank['tools'] if t['tool_name'] in missing_names]
            if gold['unavailable_decisions']!=missing:raise ValueError('unknown argument supervision changed')
            entries=[target['primary'],*target['additional_annotations']]
            if any(e['name'] not in tools for e in entries):raise ValueError('source tool outside full registry')
            if missing:
                if target['full_calls'] is not None:raise ValueError('unknown arguments falsely complete')
            else:
                direct=[dict(name=e['name'],arguments=e['arguments']) for e in entries]
                if typed(direct)!=typed(target['full_calls']):raise ValueError('complete source sequence differs')
                complete+=1
            unavailable+=len(missing);counts[row['tier']]+=1;ids.add(row['id']);unique.add(digest(native['input']))
    expected_available=dict(noul=304*len(ids),typed_call_list=152*len(ids)-unavailable,typed_tool_sequence=len(ids))
    if (len(ids)!=30733 or len(unique)!=30731 or complete!=25969
            or manifest['records']!=len(ids) or manifest['decisions_per_record']!=457 or manifest['decisions']!=457*len(ids)
            or manifest['counts_by_tier']!={str(k):v for k,v in counts.items()}
            or manifest['available_decisions_by_type']!=expected_available
            or manifest['unavailable_decisions_by_type']!={'typed_call_list':unavailable}
            or manifest['complete_argument_records']!=complete or manifest['incomplete_argument_records']!=len(ids)-complete):
        raise ValueError('full denominator/coverage arithmetic differs')
    return dict(status='verified',release=str(args.release),manifest_sha256=file_hash(args.release/'MANIFEST.json'),
        records=len(ids),unique_inputs=len(unique),decisions=457*len(ids),complete_argument_records=complete,
        incomplete_argument_records=len(ids)-complete,available_decisions_by_type=expected_available,
        unavailable_call_list_decisions=unavailable,verifier_sha256=file_hash(__file__),
        scope='All exported requests and compact gold independently compared with pinned native source rows; complete sequence and unavailable supervision arithmetic checked. No inference, teacher access, training or evaluation results.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('release','inputs','targets','registry','output'):p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('preserve earlier audit')
    write_once(args.output,verify(args))

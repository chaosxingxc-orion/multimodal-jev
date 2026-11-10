#!/usr/bin/env python3
"""Prepare full Tier7 decision prefixes using only pinned inventory/audit/registry."""
import argparse
import json
from pathlib import Path

from multimodal_jev.audio2tool_dialogue_input import decision_input
from multimodal_jev.audio2tool_inventory import registry
from multimodal_jev.baseline import atomic_json, file_hash


def run(args):
    for key in ('inventory','structure','registry'):
        if file_hash(getattr(args,key))!=getattr(args,key+'_hash'):
            raise ValueError('registered source changed: '+key)
    tools=registry(args.registry.read_bytes(),args.registry_hash)
    if len(tools)!=152:raise ValueError('complete independent registry required')
    inventory=[json.loads(line) for line in args.inventory.open()]
    rows=[row for row in inventory if row['tier']==7]
    structure=[json.loads(line) for line in args.structure.open()]
    index={row['directory_query_id']:row for row in structure}
    if (len(rows)!=2146 or len(index)!=2146 or len(structure)!=2146
            or {r['query_id'] for r in rows}!=set(index)):
        raise ValueError('full unique Tier7 roster required')
    prepared=[decision_input(row,args.registry_hash) for row in rows]
    if len({r['id'] for r in prepared})!=2146:raise ValueError('duplicate request')
    active=0; after=0; changed=[]
    with (args.output/'inputs.jsonl').open('x') as f:
        for record in prepared:
            evidence=index[record['input']['query_id']]
            audios=record['input']['audios']; post=record['provenance']['post_decision_audios']
            if (len(audios)!=evidence['decision_prefix_audio_count']
                    or len(post)!=evidence['post_final_user_audio_count']
                    or len(audios)+len(post)!=evidence['full_dialogue_audio_count']
                    or [a['role'] for a in audios+post]!=evidence['roles']
                    or evidence['final_user_text_matches_user_request'] is not True):
                raise ValueError('decision boundary differs from complete release structure audit')
            active+=len(audios);after+=len(post)
            if post:changed.append(record['id'])
            f.write(json.dumps(record)+'\n')
    if (active,after)!=(7830,4):raise ValueError('full audio accounting differs')
    atomic_json(args.output/'registry.json',tools)
    atomic_json(args.output/'result.json',dict(status='prepared',requests=len(prepared),
        actor_audio_segments=active,post_decision_audio_segments=after,
        modified_boundaries=changed,independent_tools=len(tools),
        input_hashes={k:file_hash(getattr(args,k)) for k in ('inventory','structure','registry')},
        inputs_sha256=file_hash(args.output/'inputs.jsonl'),registry_json_sha256=file_hash(args.output/'registry.json'),
        runner_sha256=file_hash(__file__),model_loaded=False,annotations_opened=False,
        scope='Complete Tier7 audio-only decision prefixes and independent registry; no fixed task prompt, tokenizer/engine acceptance, scoring, Jev target conversion or model inference. Prior structure audit parsed labeled TEST release; this preparer does not.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('inventory','structure','registry','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('inventory','structure','registry'):p.add_argument('--'+k+'-hash',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:run(a)
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0))

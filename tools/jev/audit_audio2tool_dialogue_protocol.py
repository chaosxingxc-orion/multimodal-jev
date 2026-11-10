#!/usr/bin/env python3
"""Audit Tier7 release structure; emit no transcripts or target values.

The source JSON contains labels and transcripts and is parsed for structure and
final-user equality. Never pass those records to the actor or teacher. This is
an external TEST protocol audit, not an answer-free input-only inventory.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath

from multimodal_jev.baseline import atomic_json, file_hash


def project(row, indexed):
    history = row['chat_history']
    if not history or any(t['role'] not in ('user', 'agent') for t in history):
        raise ValueError('nonempty user/agent dialogue required')
    paths = []
    for turn in history:
        p = PurePosixPath(turn['path'])
        if p.is_absolute() or '..' in p.parts or len(p.parts) != 3:
            raise ValueError('release-relative turn path required')
        paths.append('public/' + str(p))
    if paths != [a['path'] for a in indexed['audios']]:
        raise ValueError('ordered full dialogue inventory differs')
    if [t['role'] for t in history] != [a['role'] for a in indexed['audios']]:
        raise ValueError('release roles differ from filenames')
    if any(a['turn'] != i for i,a in enumerate(indexed['audios'])):
        raise ValueError('contiguous release turn positions required')
    user_positions = [i for i,t in enumerate(history) if t['role']=='user']
    if not user_positions:
        raise ValueError('missing user turn')
    final_user = user_positions[-1]
    return dict(release_id=row['id'], source_query_idx=row['query_idx'],
        directory_query_id=indexed['query_id'], roles=[t['role'] for t in history],
        final_user_position=final_user,
        final_user_text_matches_user_request=(history[final_user]['content']==row['user_request']),
        full_dialogue_audio_count=len(paths), decision_prefix_audio_count=final_user+1,
        post_final_user_audio_count=len(paths)-final_user-1,
        adjacent_same_role_positions=[i for i in range(1,len(history))
                                      if history[i]['role']==history[i-1]['role']],
        directory_matches_release_id=(indexed['query_id']==f"query_{row['id']:05d}"),
        directory_matches_source_query_idx=(indexed['query_id']==f"query_{row['query_idx']:05d}"))


def audit(args):
    if file_hash(args.manifest)!=args.manifest_hash or file_hash(args.inventory)!=args.inventory_hash:
        raise ValueError('registered manifest/inventory changed')
    manifest=json.loads(args.manifest.read_text())
    key='datasets/audio2tool/public/tier7_multiturn_data/tier7_multiturn.json'
    entry=[r for r in manifest['files'] if r['path']==key]
    if len(entry)!=1:raise ValueError('unique pinned release file required')
    entry=entry[0]; data=args.release.read_bytes()
    expected_url='https://huggingface.co/datasets/RVtech/Audio2Tool/resolve/f1388da9a3189541ab82adac88824a0661670c43/public/tier7_multiturn_data/tier7_multiturn.json'
    value=(hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
           if entry['algorithm']=='git-sha1' else hashlib.sha256(data).hexdigest())
    if (entry['algorithm'] not in ('git-sha1','sha256') or entry['url']!=expected_url
            or entry['size']!=len(data) or value!=entry['hash']):
        raise ValueError('release bytes differ from manifest')
    rows=json.loads(data)
    inventory=[json.loads(l) for l in args.inventory.open()]
    index={r['query_id']:r for r in inventory if r['tier']==7}
    if (len(rows)!=2146 or len(index)!=2146 or len({r['id'] for r in rows})!=2146
            or len({r['query_idx'] for r in rows})!=2146):
        raise ValueError('complete unique Tier7 roster required')
    records=[]
    for row in rows:
        dirs={PurePosixPath(t['path']).parts[-2] for t in row['chat_history']}
        if len(dirs)!=1:raise ValueError('dialogue spans query directories')
        records.append(project(row,index[next(iter(dirs))]))
    if {r['directory_query_id'] for r in records}!=set(index):
        raise ValueError('incomplete inventory coverage')
    with (args.output/'structure.jsonl').open('x') as f:
        for record in records:f.write(json.dumps(record)+'\n')
    report=dict(status='structure_audited',dialogues=len(records),
        audio_files=sum(r['full_dialogue_audio_count'] for r in records),
        role_patterns=dict(Counter('/'.join(r['roles']) for r in records)),
        final_user_text_matches=sum(r['final_user_text_matches_user_request'] for r in records),
        directory_release_id_matches=sum(r['directory_matches_release_id'] for r in records),
        directory_source_query_idx_matches=sum(r['directory_matches_source_query_idx'] for r in records),
        post_final_user_dialogues=[r for r in records if r['post_final_user_audio_count']],
        adjacent_same_role_dialogues=[r for r in records if r['adjacent_same_role_positions']],
        distinct_instruction_count=len({r['instruction'] for r in rows}),
        release_sha256=file_hash(args.release),manifest_sha256=args.manifest_hash,
        inventory_sha256=args.inventory_hash,structure_sha256=file_hash(args.output/'structure.jsonl'),
        runner_sha256=file_hash(__file__),model_loaded=False,actor_or_teacher_inputs_created=False,
        scope='Full TEST release JSON parsed for structural audit and final-user text equality; labels/transcripts reside in source, no target values or transcripts emitted, no scoring/prompt selection. Final decision-prefix protocol remains a separate explicit choice.')
    atomic_json(args.output/'result.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('release','manifest','inventory','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('manifest','inventory'):p.add_argument('--'+k+'-hash',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:audit(a)
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0))

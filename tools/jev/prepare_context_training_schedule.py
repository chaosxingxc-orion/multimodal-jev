#!/usr/bin/env python3
"""Build a uniform-query full schedule from authenticated clean learning data."""
import argparse
from collections import Counter
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_training_schedule import build_epoch
from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.slurp_official_training import load


def prepare(release,release_sha,ledger,ledger_sha,output):
    if output.exists():raise FileExistsError('immutable schedule exists')
    if file_hash(ledger)!=ledger_sha:raise ValueError('processor ledger changed')
    rows,task,_=load(release,release_sha,'learning',purpose='optimize')
    population={r['id']:r for r in rows}
    candidates=[dict(id=c['id'],definition=c['description']) for c in task['schema']['candidates']]
    lengths={};requests={}
    for line in ledger.read_text().splitlines():
        row=json.loads(line)
        if row['role']!='learning':continue
        source=population[row['id']]
        request=compile_candidates(source['audio'],candidates,task['thinking']['requirements'],semantics='exclusive_event')
        if (row['id'] in lengths or row['audio_sha256']!=source['audio_sha256'] or
                row['source_group']!=source['source_group'] or row['request_sha256']!=content_hash(request.model_input()) or
                not row['fits_context'] or not 0<row['tokens']<=4096 or len(row['positions'][0])!=60):
            raise ValueError('complete new context processor/input identity required')
        # Ledger stores padded processor feature width, not valid frame count.
        # Use it only for schedule grouping; execution validates actual masks.
        lengths[row['id']]=(row['tokens'],row['input_features']['shape'][-1])
        requests[row['id']]=row['request_sha256']
    if set(lengths)!=set(population):raise ValueError('full learning ledger required')
    epochs=[build_epoch(rows,lengths,epoch=e) for e in range(2)]
    reports=[]
    output.mkdir(parents=True)
    for epoch,value in enumerate(epochs):
        p=output/f'epoch-{epoch}.json';p.write_text(json.dumps(value,separators=(',',':'))+'\n')
        # Independent population and rank reconstruction, not just manifest fields.
        real=[slot for batch in value['batches'] for slot in batch['slots'] if not slot['padding']]
        assert Counter(s['id'] for s in real)==Counter(population.keys())
        assert all(s['weight']==1. for s in real)
        windows=[]
        for start in (0,100,200,285):
            subset=value['batches'][start:start+32]
            counts=Counter(population[s['id']]['intent'] for b in subset for s in b['slots'] if not s['padding'])
            windows.append(dict(start=start,updates=len(subset),classes=len(counts)))
        reports.append(dict(value['report'],file_sha256=file_hash(p),windows=windows))
    manifest=dict(schema='context-full-learning-schedule-v1',release=str(release),release_sha256=release_sha,
        ledger_sha256=ledger_sha,request_identity_sha256=content_hash(requests),epochs=reports,
        stage_reuse='same_full_epoch_memberships_for_instruction_and_task_stages',
        global_batch=128,world_size=4,microbatch=8,accumulation=4,total_updates_per_stage=sum(x['updates'] for x in reports),
        annotation_semantics='exclusive_event',labels_in_question_selection=False,oversampling=False,
        grouping='within_global_batch_by_tokens_and_processor_feature_width',training_dispatch_allowed=False,
        input_scope='Reuses authenticated context compiler ledger; not old Final60 input acceptance or NPU admission')
    (output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--release-sha',required=True)
    p.add_argument('--ledger',type=Path,required=True);p.add_argument('--ledger-sha',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();m=prepare(a.release,a.release_sha,a.ledger,a.ledger_sha,a.output)
    print(json.dumps(dict(updates=m['total_updates_per_stage'],epochs=[dict(recordings=x['recordings'],padding=x['padding_slots'],windows=x['windows']) for x in m['epochs']]),indent=2))

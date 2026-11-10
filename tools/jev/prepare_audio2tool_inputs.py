#!/usr/bin/env python3
"""All release-record inputs from audited projections, without annotation access."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

from multimodal_jev.audio2tool_input import REGISTRY_SHA256,system_text,validate
from multimodal_jev.audio2tool_inventory import REVISION
from multimodal_jev.baseline import atomic_json,file_hash


def build(inventory,mapping,dialogues,structure):
    audio={a['path']:a for row in inventory for a in row['audios']}
    if len(audio)!=sum(len(row['audios']) for row in inventory):raise ValueError('duplicate inventory path')
    result=[]
    for row in mapping:
        if set(row)!={'tier','release_id','source_query_idx','audio_path','audio_sha256'}:
            raise ValueError('strict annotation-free release mapping required')
        if row['tier'] not in (1,2,3,4,5,6,8):raise ValueError('single audio tier required')
        a=audio[row['audio_path']]
        if a['sha256']!=row['audio_sha256'] or not a['path'].startswith(f"public/tier{row['tier']}_"):
            raise ValueError('mapped audio identity differs')
        result.append(dict(id=f"audio2tool:tier{row['tier']}:release:{row['release_id']}",
            tier=row['tier'],release_id=row['release_id'],source_query_idx=row['source_query_idx'],
            input=dict(mode='foreground_mixture' if row['tier']==8 else 'single_audio',
                       registry_sha256=REGISTRY_SHA256,audios=[dict(a)]),
            provenance=dict(post_decision_audios=[])))
    index={r['directory_query_id']:r for r in structure}
    if len(index)!=len(structure):raise ValueError('duplicate structure identity')
    for row in dialogues:
        if row['schema']!='audio2tool-tier7-decision-input-v1' or row['revision']!=REVISION or row['split']!='external_test':
            raise ValueError('registered Tier7 input required')
        inp=row['input'];audit=index[inp['query_id']]
        active=inp['audios'];post=row['provenance']['post_decision_audios']
        original=next(r['audios'] for r in inventory if r['tier']==7 and r['query_id']==inp['query_id'])
        if (active+post!=original or len(active)!=audit['decision_prefix_audio_count']
                or len(post)!=audit['post_final_user_audio_count'] or inp['registry_sha256']!=REGISTRY_SHA256):
            raise ValueError('Tier7 prefix/provenance changed')
        result.append(dict(id=f"audio2tool:tier7:release:{audit['release_id']}",tier=7,
            release_id=audit['release_id'],source_query_idx=audit['source_query_idx'],
            input=dict(mode='dialogue_last_user',registry_sha256=REGISTRY_SHA256,audios=active),
            provenance=dict(post_decision_audios=post)))
    for row in result:
        validate(row['input'])
        row.update(schema='audio2tool-native-audio-input-v1',revision=REVISION,split='external_test')
    if len({r['id'] for r in result})!=len(result):raise ValueError('duplicate release identity')
    return sorted(result,key=lambda r:(r['tier'],r['release_id']))


def run(args):
    keys=('inventory','mapping','dialogues','structure','registry')
    for k in keys:
        if file_hash(getattr(args,k))!=getattr(args,k+'_hash'):raise ValueError('registered source changed: '+k)
    def rows(k):return [json.loads(l) for l in getattr(args,k).open()]
    text=system_text(args.registry.read_bytes())
    records=build(*(rows(k) for k in keys[:-1]))
    if len(records)!=30733 or Counter(r['tier'] for r in records)!={1:4292,2:6320,3:4292,4:4278,5:4292,6:4292,7:2146,8:821}:
        raise ValueError('full release-record denominator required')
    grouped=defaultdict(list)
    for r in records:
        pin=hashlib.sha256(json.dumps(r['input'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        grouped[pin].append(r['id'])
    if len(grouped)!=30731:raise ValueError('unique input denominator differs')
    with (args.output/'inputs.jsonl').open('x') as f:
        for row in records:f.write(json.dumps(row)+'\n')
    (args.output/'system.txt').write_text(text+'\n')
    atomic_json(args.output/'result.json',dict(status='prepared',requests=len(records),unique_inputs=len(grouped),
        counts_by_tier=dict(Counter(r['tier'] for r in records)),duplicate_input_groups=[ids for ids in grouped.values() if len(ids)>1],
        actor_audio_references=sum(len(r['input']['audios']) for r in records),
        post_decision_audio_references=sum(len(r['provenance']['post_decision_audios']) for r in records),
        input_hashes={k:file_hash(getattr(args,k)) for k in keys},inputs_sha256=file_hash(args.output/'inputs.jsonl'),
        system_file_sha256=file_hash(args.output/'system.txt'),runner_sha256=file_hash(__file__),
        model_loaded=False,annotations_opened=False,predictions_reused=False,
        scope='All release-record actor inputs; fixed local JSON protocol, complete independent registry. No official paper prompt equivalence, full context/engine/scorer acceptance or Jev target conversion.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('inventory','mapping','dialogues','structure','registry','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('inventory','mapping','dialogues','structure','registry'):p.add_argument('--'+k+'-hash',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:run(a)
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0))

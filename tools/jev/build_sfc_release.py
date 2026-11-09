#!/usr/bin/env python3
"""Freeze all official SFC splits and multi-turn inputs without changing labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from multimodal_jev.baseline import file_hash, atomic_json
from multimodal_jev.sfc import parse_calls, source_identity, model_input


def build(root, destination):
    destination.mkdir(parents=True, exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.sfc-',dir=destination))
    sources={};audio_hashes={};splits={}
    try:
        for split in ('train_id','test_id','test_ood'):
            path=root/'data/processed/data_msswift'/f'{split}.json'
            sources[str(path.relative_to(root))]=file_hash(path)
            rows=json.loads(path.read_text());seen=set();groups=set()
            with (stage/(split+'.jsonl')).open('w') as out:
                for row in rows:
                    identity=source_identity(row['audios'])
                    if identity['official_split']!=split:raise ValueError('split mismatch')
                    identifier='sfc:'+row['audios'][-1]
                    if identifier in seen:raise ValueError('duplicate example')
                    seen.add(identifier);groups.add(identity['source_group'])
                    inputs=model_input(row)
                    if inputs['messages'][-1]['role']!='user':raise ValueError('answer in prompt')
                    if sum(m['content'].count('<audio>') for m in inputs['messages'])!=len(inputs['audios']):
                        raise ValueError('audio turn mismatch')
                    paths=[];hashes=[]
                    for relative in inputs['audios']:
                        if source_identity([relative])['source_group']!=identity['source_group']:
                            raise ValueError('cross-conversation audio')
                        audio=root/'data'/relative
                        if relative not in audio_hashes:audio_hashes[relative]=file_hash(audio)
                        paths.append(str(audio));hashes.append(audio_hashes[relative])
                    inputs['audios']=paths
                    record={'example_id':identifier,'input':inputs,
                            'supervision':{'official_solution':row['solution'],'calls':parse_calls(row['solution'])},
                            'metadata':{**identity,'audio_sha256':hashes,'relative_audios':row['audios'],
                                        'source_dataset':'SFC-Bench','audio_source':'TTS',
                                        'history_policy':'official-supplied-assistant-feedback; not model-generated'}}
                    out.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+'\n')
            splits[split]={'records':len(rows),'source_groups':len(groups),'sha256':file_hash(stage/(split+'.jsonl'))}
        receipt={'protocol':'sfc-official-v1','sources_sha256':sources,'splits':splits,
                 'audio_files':len(audio_hashes),'audio_inventory_sha256':hashlib.sha256(json.dumps(audio_hashes,sort_keys=True).encode()).hexdigest(),
                 'builder_sha256':file_hash(__file__),
                 'parser_sha256':file_hash(Path(__file__).parents[2]/'src/multimodal_jev/sfc.py'),
                 'scope':'official split preservation; train is reference only; test-ID/OOD held out from FSC/SLURP training; offline supplied history, not interactive execution'}
        release=hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest()
        receipt['release_id']=release
        final=destination/release
        if final.exists():raise FileExistsError(final)
        atomic_json(stage/'audio-hashes.json',audio_hashes)
        atomic_json(stage/'MANIFEST.json',receipt)
        os.rename(stage,final)
        return final,receipt
    except BaseException:
        shutil.rmtree(stage)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
    path,receipt=build(a.root,a.output_root)
    print(json.dumps({'path':str(path),**receipt},indent=2))

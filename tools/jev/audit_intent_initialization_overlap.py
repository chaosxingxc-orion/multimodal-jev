#!/usr/bin/env python3
"""Audit fixed development against initialization and RL exposure; no scores."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.policy_identity import checkpoint_identity
from audit_intent_development_overlap import compare,run as audit_pilot
from audit_intent_development_audio import inspect_audio


def training_population(bundles,balance,history,result):
    from multimodal_jev.span_balance import balanced_binary_batches
    _,expected=balanced_binary_batches(bundles,balance['seed'])
    if expected!=balance:raise ValueError('reconstructed initialization schedule differs')
    schedule=balance['batch_recording_ids']
    if len(history)!=len(schedule) or result['updates']!=len(schedule):raise ValueError('complete actual initialization history required')
    for index,(row,ids) in enumerate(zip(history,schedule,strict=True),1):
        if row['step']!=index or row['recordings']!=ids:raise ValueError('actual initialization sources differ from schedule')
    used={uid for ids in schedule for uid in ids}
    if result['recordings']!=len(bundles) or result['training_recordings']!=len(used):raise ValueError('initialization population counts differ')
    return [b for b in bundles if b['id'] in used]


def union_records(*populations):
    by_id={}
    for rows in populations:
        for row in rows:
            # Task suffixes differ; audio/source identity defines exposure.
            key=(row['source_group'],row['file'],row['audio_sha256'])
            if key not in by_id:by_id[key]=dict(row,id='exposure:'+str(len(by_id)))
    return list(by_id.values())


def run(initialization,gate,pilot,manifest_sha256,development,speaker_metadata):
    from multimodal_jev.span_training import load_reviewed_bundles
    from queue_span_native_actor_npu import completed
    from queue_span_dp4_resume import check_pins
    initialization,gate,development=map(Path,(initialization,gate,development))
    read=lambda p:json.loads(p.read_text())
    config=read(initialization/'config.json');launch=read(initialization/'launch.json')
    if not completed(launch['container_id'],launch['image']):raise ValueError('actual initialization completion required')
    check_pins({str(initialization/'config.json'):launch['config_sha256'],str(initialization/'source-manifest.json'):launch['source_manifest_sha256']})
    check_pins(config['pins'])
    source_manifest=read(initialization/'source-manifest.json')
    check_pins({str(initialization/'source'/p):h for p,h in source_manifest['files'].items()})
    result=read(initialization/'run/result.json')
    if read(initialization/'run/terminal.json')['exit_code']!=0 or result['status']!='passed':raise ValueError('successful initialization required')
    adapter=initialization/'run/adapter';adapter_id,adapter_files=checkpoint_identity(adapter)
    for name,h in result['adapter_files'].items():
        if file_hash(adapter/name)!=h:raise ValueError('initialization adapter changed')
    gc=read(gate/'config.json');gl=read(gate/'launch.json')
    check_pins({str(gate/'config.json'):gl['config_sha256']})
    if gc['adapter']!=str(adapter) or gc['adapter_id']!=adapter_id:raise ValueError('RL start does not use audited initialization')
    bundles,_=load_reviewed_bundles(config['release'],config['manifest_sha256'],config['question_order'])
    if config['sampling_mode']!='joint_uniform_complement_pairs_v1':raise ValueError('explicit initialization schedule required')
    balance=read(initialization/'run/balance.json');history=read(initialization/'run/history.json')
    if balance['seed']!=config['seed']:raise ValueError('initialization seed differs')
    trained=training_population(bundles,balance,history,result)
    def rows(items):return [dict(id=b['id'],file=Path(b['request'].audio).name,source_group=b['source_group'],audio_sha256=b['audio_sha256']) for b in items]
    original=audit_pilot(pilot,manifest_sha256,development,speaker_metadata)
    rl=[{k:v for k,v in r.items() if k!='speaker'} for r in original['records']['learning']]
    dev=[{k:v for k,v in r.items() if k!='speaker'} for r in original['records']['development']]
    populations={'initialization_trained':rows(trained),'initialization_diagnostic':rows(bundles),
                 'rl_learning':rl,'all_weight_training':union_records(rows(trained),rl),
                 'all_known_exposure':union_records(rows(bundles),rl)}
    wanted={r['file'] for rs in [dev,*populations.values()] for r in rs};speakers={}
    for source in read(Path(speaker_metadata)).values():
        for filename,record in source['recordings'].items():
            if filename in wanted:
                speaker=record.get('usrid')
                if not isinstance(speaker,str) or not speaker.strip():raise ValueError('publisher speaker missing')
                speakers.setdefault(filename,set()).add(speaker)
    comparisons={name:compare(rs,dev,speakers) for name,rs in populations.items()}
    # Decode the exact union only once. Identical PCM can survive FLAC metadata changes.
    audio_paths={Path(b['request'].audio).name:b['request'].audio for b in bundles}
    from multimodal_jev.slurp_intent_pilot import load_pilot
    items,_=load_pilot(pilot,manifest_sha256,'D')
    audio_paths.update({Path(i['request'].audio).name:i['request'].audio for i in items})
    audio_paths.update({Path(r['request']['input']['audio']).name:r['request']['input']['audio'] for r in read(development/'packet.json')})
    signals={}
    for r in dev+populations['all_known_exposure']:
        key=r['file']
        if key not in signals:signals[key]=inspect_audio(audio_paths[key],r['audio_sha256'])
        elif signals[key]['audio_sha256']!=r['audio_sha256']:raise ValueError('ambiguous filename identity')
    pcm={}
    for name,rs in populations.items():
        left={signals[r['file']]['waveform_sha256'] for r in rs}
        matches=[dict(id=r['id'],file=r['file'],waveform_sha256=signals[r['file']]['waveform_sha256']) for r in dev if signals[r['file']]['waveform_sha256'] in left]
        pcm[name]=dict(development_recordings=len(matches),matches=matches)
    files=dict(original['files'])
    for root,names in [(initialization,['config.json','launch.json','source-manifest.json','run/result.json','run/history.json','run/balance.json','run/terminal.json']),
                       (gate,['config.json','launch.json'])]:
        files.update({str(root/n):file_hash(root/n) for n in names})
    files.update({str(adapter/n):h for n,h in adapter_files.items()})
    for path in [Path(__file__),Path(config['release'])/'MANIFEST.json',Path(config['release'])/'train.jsonl']:
        files[str(path.resolve())]=file_hash(path)
    return dict(initialization_adapter_id=adapter_id,initialization_container_id=launch['container_id'],
        comparisons=comparisons,decoded_audio_files=len(signals),pcm_overlap=pcm,files=files,
        evaluation_allowed=False,optimization_allowed=False,population_changed=False,
        scope='Known temporal initialization and current intent RL exposure only. Exact source/file/bytes/PCM and publisher speakers; no near-duplicate, pretraining, semantic listening or benchmark admission claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('initialization','gate','pilot','manifest-sha256','development','speaker-metadata'):p.add_argument('--'+name,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable exposure audit exists')
    atomic_json(a.output,run(a.initialization,a.gate,a.pilot,a.manifest_sha256,a.development,a.speaker_metadata))

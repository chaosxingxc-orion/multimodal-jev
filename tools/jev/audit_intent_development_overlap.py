#!/usr/bin/env python3
"""Measure fixed learning/development overlap without labels or reselection."""
import argparse
from collections import Counter
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.slurp_intent_pilot import load_pilot


def compare(learning,development,speakers):
    populations={}
    for name,rows in [('learning',learning),('development',development)]:
        if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('unique nonempty population required')
        records=[]
        for row in rows:
            owners=speakers.get(row['file'],set())
            if len(owners)!=1:raise ValueError('unique publisher speaker required for '+row['file'])
            records.append(dict(row,speaker=next(iter(owners))))
        populations[name]=records
    a,b=populations['learning'],populations['development']
    sets=lambda rows,field:{r[field] for r in rows}
    overlap={field:sorted(sets(a,field)&sets(b,field)) for field in ('source_group','file','audio_sha256','speaker')}
    seen=sets(a,'speaker');count=Counter(r['speaker'] for r in b)
    return dict(counts={k:dict(recordings=len(v),source_groups=len(sets(v,'source_group')),
        speakers=len(sets(v,'speaker'))) for k,v in populations.items()},
        overlap_counts={k:len(v) for k,v in overlap.items()},overlaps=overlap,
        development_seen_speaker_recordings=sum(count[s] for s in seen),
        development_unseen_speaker_recordings=sum(n for s,n in count.items() if s not in seen),
        development_speaker_counts=dict(sorted(count.items())),records=populations)


def run(pilot,manifest_sha256,development,speaker_metadata):
    development=Path(development);speaker_metadata=Path(speaker_metadata)
    items,admission=load_pilot(pilot,manifest_sha256,'D')
    report=json.loads((development/'report.json').read_text())
    if file_hash(development/'packet.json')!=report['packet_sha256']:raise ValueError('fixed development packet changed')
    for path,digest in report['evidence_files'].items():
        if file_hash(path)!=digest:raise ValueError('development provenance changed')
    packet=json.loads((development/'packet.json').read_text())
    if len(packet)!=report['recordings']:raise ValueError('complete fixed development population required')
    learning=[dict(id=i['id'],source_group=i['target']['source_group'],file=Path(i['request'].audio).name,
                   audio_sha256=i['audio_sha256']) for i in items]
    dev=[]
    for source in packet:
        r=source['request']
        if r['role']!='selection' or r['official_split']!='train':raise ValueError('fixed TRAIN selection only')
        if source['evaluation_allowed'] is not False or source['optimization_allowed'] is not False:
            raise ValueError('pending population only')
        if file_hash(r['input']['audio'])!=r['audio_sha256']:raise ValueError('development audio changed')
        dev.append(dict(id=r['id'],source_group=r['source_group'],file=Path(r['input']['audio']).name,audio_sha256=r['audio_sha256']))
    # Read publisher recording metadata only for requested filenames; no transcript or model outputs.
    wanted={r['file'] for r in learning+dev};speakers={}
    for source in json.loads(speaker_metadata.read_text()).values():
        for filename,record in source['recordings'].items():
            if filename not in wanted:continue
            speaker=record.get('usrid')
            if type(speaker)!=str or not speaker.strip():raise ValueError('publisher speaker missing')
            speakers.setdefault(filename,set()).add(speaker)
    result=compare(learning,dev,speakers)
    files=[Path(pilot)/'MANIFEST.json',development/'report.json',development/'packet.json',speaker_metadata,Path(__file__).resolve()]
    return dict(result,pilot_admission=admission,files={str(p.resolve()):file_hash(p) for p in files},
        model_predictions_used=False,population_changed=False,evaluation_allowed=False,optimization_allowed=False,
        scope='Publisher speaker IDs and exact file/hash/source overlap only; no waveform near-duplicate detection, semantic listening admission, model evaluation or speaker identity re-verification.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('pilot','manifest-sha256','development','speaker-metadata'):p.add_argument('--'+name,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable audit exists')
    atomic_json(a.output,run(a.pilot,a.manifest_sha256,a.development,a.speaker_metadata))

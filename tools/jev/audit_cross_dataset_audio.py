#!/usr/bin/env python3
"""Input-only exact encoded-audio hash overlap across pinned native releases.

No target files, waveform decoding or task-score computation. This compares
previously recorded byte hashes, not acoustic/semantic near duplicates.
"""
import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

from multimodal_jev.baseline import atomic_json,file_hash


def compare(cells):
    result=[]
    for a,b in combinations(sorted(cells),2):
        common=set(cells[a]) & set(cells[b])
        result.append(dict(left=a,right=b,shared_audio_hashes=len(common),
            left_affected_requests=len({x for h in common for x in cells[a][h]}),
            right_affected_requests=len({x for h in common for x in cells[b][h]})))
    return result


def audit(plan):
    if plan['schema']!='native-audio-overlap-plan-v1' or not plan['datasets']:
        raise ValueError('explicit native release plan required')
    cells={};counts={};sources={};names=set()
    for item in plan['datasets']:
        name=item['dataset'];root=Path(item['release']);manifest=root/'MANIFEST.json'
        if name in names:raise ValueError('duplicate dataset identity')
        names.add(name)
        if file_hash(manifest)!=item['manifest_sha256']:raise ValueError('registered manifest changed')
        data=json.loads(manifest.read_text())
        native_slurp = name=='slurp-native' and data['identity'].get('schema')=='slurp-native-release-v1'
        if (not native_slurp and data['identity'].get('dataset')!=name) or data['release_id']!=root.name:
            raise ValueError('native release identity differs')
        sources[name]=dict(release=str(root),manifest_sha256=item['manifest_sha256'],requests={})
        for split,info in sorted(data['splits'].items()):
            path=root/(split+'.requests.jsonl')
            if file_hash(path)!=info['requests_sha256']:raise ValueError('registered requests changed')
            ids=set();groups=set();mapping=defaultdict(set);audio_occurrences=0
            with path.open() as handle:
                for line in handle:
                    row=json.loads(line)
                    if set(row)!={'id','input','source_group','split','audio_sha256'}:
                        raise ValueError('input-only native request schema required')
                    if row['split']!=split or row['id'] in ids:raise ValueError('wrong split or repeated request ID')
                    ids.add(row['id']);groups.add(row['source_group'])
                    hashes=row['audio_sha256']
                    hashes=[hashes] if isinstance(hashes,str) else hashes
                    paths=([row['input']['audio']] if 'audio' in row['input'] else row['input']['audios'])
                    if not hashes or len(hashes)!=len(paths):raise ValueError('aligned audio hashes required')
                    for digest in hashes:
                        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
                            raise ValueError('valid encoded-audio hash required')
                        mapping[digest].add(row['id']);audio_occurrences+=1
            if len(ids)!=info['recordings' if native_slurp else 'requests'] or len(groups)!=info['source_groups']:
                raise ValueError('complete native population required')
            cell=name+'/'+split;cells[cell]=mapping
            counts[cell]=dict(requests=len(ids),source_groups=len(groups),audio_occurrences=audio_occurrences,
                unique_audio_hashes=len(mapping),hashes_used_by_multiple_requests=sum(len(v)>1 for v in mapping.values()))
            sources[name]['requests'][split]=dict(path=str(path),sha256=info['requests_sha256'])
    return dict(schema='native-audio-overlap-audit-v1',sources=sources,counts=counts,pairs=compare(cells),
        target_files_read=False,manifest_metadata_read=True,waveforms_rehashed=False,model_called=False,
        scope='Exact encoded-file hashes inherited from pinned complete request releases. '
              'No decoded-waveform, near-duplicate, transcript, semantic or pretraining contamination claim.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-hash',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('preserve prior audit')
    if file_hash(args.plan)!=args.plan_hash:raise ValueError('registered plan changed')
    result=audit(json.loads(args.plan.read_text()));result['plan_sha256']=args.plan_hash
    result['runner_sha256']=file_hash(__file__);atomic_json(args.output,result)

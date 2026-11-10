#!/usr/bin/env python3
"""Score completed generative bundle tasks and reconstruct official native SLU."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.slurp_jev_score import score_bundle,summarize
from multimodal_jev.slurp_score import score_native,load_official_metrics


def score_run(run,release,repository,output):
    if output.exists():raise FileExistsError('completed score retained')
    identity=json.loads((run/'identity.json').read_text());cfg=identity['config']
    result=json.loads((run/'result.json').read_text())
    if (cfg['route']!='audio-direct' or cfg['parameters_frozen'] is not True
            or result['status']!='passed' or result['weight_updates']!=0 or not result['model_files_unchanged']
            or json.loads((run/'terminal.json').read_text())['exit_code']!=0):
        raise ValueError('completed frozen direct audio run required')
    if file_hash(run/'predictions.jsonl')!=result['predictions_sha256']:
        raise ValueError('predictions changed')
    requests_path=Path(cfg['requests'])
    materialized=json.loads((requests_path.parent/'MANIFEST.json').read_text())
    if (file_hash(requests_path)!=identity['requests_sha256']
            or identity['requests_sha256']!=materialized['requests_sha256']
            or file_hash(release/'MANIFEST.json')!=materialized['source_manifest_sha256']):
        raise ValueError('resolved request source changed')
    manifest=json.loads((release/'MANIFEST.json').read_text());split=cfg['split']
    if split!=materialized['split']:raise ValueError('split changed')
    bank=json.loads((release/'task-bank.json').read_text())
    if file_hash(release/'task-bank.json')!=manifest['identity']['task_bank_sha256']:
        raise ValueError('task bank changed')
    for kind in ('bundles','targets'):
        if file_hash(release/f'{split}.{kind}.jsonl')!=manifest['splits'][split][kind+'_sha256']:
            raise ValueError('Jev release changed')
    groups=set(materialized['source_groups'])
    if cfg.get('source_groups') is not None and set(cfg['source_groups'])!=groups:
        raise ValueError('score requires all materialized source groups')
    by_bundle=defaultdict(list);predictions=[]
    for line in (run/'predictions.jsonl').open():
        row=json.loads(line);by_bundle[row['example_id'].rsplit('::',1)[0]].append(row)
        predictions.append(row)
    selected=json.loads((run/'selection.json').read_text())['ids']
    request_ids=[json.loads(line)['id'] for line in requests_path.open()]
    if (selected!=request_ids or len(selected)!=len(set(selected))
            or set(selected)!={r['example_id'] for r in predictions}
            or len(predictions)!=len(selected) or result['predictions']!=len(predictions)):
        raise ValueError('complete materialized request coverage required')
    targets={r['id']:r for r in map(json.loads,(release/f'{split}.targets.jsonl').open())}
    native=Path(manifest['identity']['native_release'])
    if file_hash(native/'MANIFEST.json')!=manifest['identity']['native_manifest_sha256']:
        raise ValueError('native source changed')
    native_manifest=json.loads((native/'MANIFEST.json').read_text())
    for kind in ('requests','targets'):
        if file_hash(native/f'{split}.{kind}.jsonl')!=native_manifest['splits'][split][kind+'_sha256']:
            raise ValueError('native source records changed')
    native_requests={r['id']:r for r in map(json.loads,(native/f'{split}.requests.jsonl').open())}
    native_targets={r['id']:r for r in map(json.loads,(native/f'{split}.targets.jsonl').open())}
    counts=Counter();seen=set();reconstructed=[];native_selected=[]
    for line in (release/f'{split}.bundles.jsonl').open():
        bundle=json.loads(line)
        if bundle['source_group'] not in groups:continue
        uid=bundle['id'];target=targets[uid]
        if uid in seen:raise ValueError('duplicate bundle')
        if hashlib.sha256(canonical(native_targets[uid])).hexdigest()!=target['native_target_sha256']:
            raise ValueError('native target binding changed')
        cell,output_native=score_bundle(bundle,target,bank,by_bundle.get(uid,[]));counts.update(cell)
        reconstructed.append({'example_id':uid,'replicate':0,
            'request_sha256':hashlib.sha256(canonical(native_requests[uid])).hexdigest(),
            'raw_text':json.dumps(output_native) if output_native is not None else 'invalid task bundle'})
        native_selected.append(native_requests[uid]);seen.add(uid)
    if seen!=set(by_bundle) or len(seen)!=materialized['recordings']:
        raise ValueError('complete selected bundle coverage required')
    official,normalized=score_native(native_selected,list(native_targets.values()),reconstructed,
        json.loads((native/'taxonomy.json').read_text()),load_official_metrics(repository))
    report=summarize(counts)
    report.update(official_native_reconstruction=official,run=str(run),release=str(release),
        split=split,full_official_split=counts['recordings']==manifest['splits'][split]['recordings'],
        result_sha256=file_hash(run/'result.json'))
    output.mkdir(parents=True)
    atomic_json(output/'result.json',report)
    (output/'official-predictions.jsonl').write_bytes(b''.join(map(canonical,normalized)))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('run','release','repository','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();r=score_run(a.run,a.release,a.repository,a.output)
    print(json.dumps({k:v for k,v in r.items() if k!='official_native_reconstruction'},indent=2))

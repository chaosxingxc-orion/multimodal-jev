#!/usr/bin/env python3
"""Accept exact complete Jev TRAIN generation, then score unchanged raw outputs."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.consumption import canonical
from infer_sfc_jev_omni_probe import select_inputs, digest, SOURCE_FILES
from watch_qwen3_omni_probe import inspect
from evaluate_sfc_jev import evaluate


def accept(root):
    cfg=json.loads((root/'config.json').read_text());rows,preflight,checks=select_inputs(cfg)
    plan=json.loads((root/'preparation.json').read_text());launch=json.loads((root/'launch.json').read_text())
    item=inspect(launch['container_id'])
    if (item['Id']!=launch['container_id'] or item['Image']!=plan['image']
            or item['State']['Status']!='exited' or item['State']['ExitCode']!=0
            or launch['source_commit']!=plan['source_commit'] or launch['physical_devices']!=[6,7]
            or file_hash(root/'config.json')!=plan['config_sha256']):
        raise ValueError('exact successful frozen container required')
    run=root/'run';result=json.loads((run/'result.json').read_text());identity=json.loads((run/'identity.json').read_text())
    audit=json.loads(Path(cfg['asset_audit']).read_text())
    if (json.loads((run/'terminal.json').read_text())['exit_code']!=0 or result['status']!='passed'
            or result['weight_updates']!=0 or result['model_files_unchanged'] is not True
            or identity['config']!=cfg or identity['model_files']!=audit['files']
            or identity['versions']!=cfg['versions'] or identity['inputs_sha256']!=digest(rows)
            or identity['processor_preflight_sha256']!=cfg['processor_preflight_sha256']
            or file_hash(run/'predictions.jsonl')!=result['predictions_sha256']):
        raise ValueError('frozen run identity changed')
    if set(identity['source_files'])!=set(SOURCE_FILES):raise ValueError('complete source custody required')
    for name,value in identity['source_files'].items():
        if file_hash(root/'source'/name)!=value or plan['inference_files'][name]!=value:
            raise ValueError('inference source changed')
    predictions=[json.loads(line) for line in (run/'predictions.jsonl').open()]
    if (len(predictions)!=78 or result['decisions']!=78 or result['predictions']!=78 or result['bundles']!=6
            or [(p['example_id'],p['decision_id']) for p in predictions]!=list(checks)
            or json.loads((run/'selection.json').read_text())!=dict(ids=[r['id'] for r in rows],decisions=[list(k) for k in checks])):
        raise ValueError('all original 78 decisions required')
    by_id={r['id']:r for r in rows}
    for p in predictions:
        row=by_id[p['example_id']];check=checks[p['example_id'],p['decision_id']]
        if (p['request_sha256']!=digest(row) or p['audio_sha256']!=row['audio_sha256']
                or p['split']!='train_id' or p['source_group']!=row['source_group']
                or p['hf_vllm_prompt_ids_equal'] is not True
                or any(p[k]!=check[k] for k in ('input_tokens','input_ids_sha256','prompt_sha256','expected_frames'))
                or type(p['raw_text']) is not str or len(p['token_ids'])>1024
                or p['finish_reason'] not in ('stop','length')
                or type(p['length_limited']) is not bool or p['length_limited']!=(p['finish_reason']=='length')):
            raise ValueError('prediction input binding or engine parity changed')
    if result['length_limited']!=sum(p['length_limited'] for p in predictions):raise ValueError('termination count changed')
    return rows,preflight,predictions


def score(root):
    rows,preflight,predictions=accept(root)
    plan=json.loads((root/'preparation.json').read_text());release=Path(preflight['release'])
    manifest=json.loads((release/'MANIFEST.json').read_text());source=release/'train_id.targets.jsonl'
    if file_hash(source)!=manifest['splits']['train_id']['targets_sha256']:raise ValueError('targets changed')
    wanted={r['id'] for r in rows};targets=[json.loads(line) for line in source.open() if json.loads(line)['id'] in wanted]
    if [r['id'] for r in targets]!=[r['id'] for r in rows]:raise ValueError('exact TRAIN targets required')
    subset=root/'scoring-subset';subset.mkdir(exist_ok=False)
    pin=dict(bundles=6,decisions=78)
    for kind,values in (('requests',rows),('targets',targets)):
        path=subset/f'train_id.{kind}.jsonl';path.write_bytes(b''.join(map(canonical,values)));pin[kind+'_sha256']=file_hash(path)
    (subset/'MANIFEST.json').write_bytes(canonical(dict(source_manifest_sha256=file_hash(release/'MANIFEST.json'),
        scope='Exact processor-selected six TRAIN turns only; no full-split performance claim.',splits={'train_id':pin})))
    path=root/'scoring-predictions.jsonl'
    path.write_bytes(b''.join(canonical({k:p[k] for k in ('example_id','decision_id','request_sha256','raw_text')}) for p in predictions))
    report=evaluate(subset,'train_id',path,Path(plan['official_scorer']),root/'score',plan['official_scorer_sha256'])
    atomic_json(root/'acceptance.json',dict(bundles=6,decisions=78,source_groups=4,weight_updates=0,
        counts=report['counts'],metrics_sha256=file_hash(root/'score/metrics.json'),
        scope='Complete TRAIN engineering probe; no quality threshold or TEST claim.'))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    print(json.dumps(score(a.root),indent=2))

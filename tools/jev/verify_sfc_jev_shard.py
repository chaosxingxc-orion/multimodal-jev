#!/usr/bin/env python3
"""Accept an entire generated shard without reading targets or scoring quality."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_jev_shard_runtime import select_inputs,check_probe,check_predictions,digest,read
from infer_sfc_jev_omni_shard import SOURCE_FILES
from verify_sfc_jev_probe import accept as accept_probe
from verify_sfc_jev_context import verify as verify_context
from watch_qwen3_omni_probe import inspect
from multimodal_jev.sfc_candidate_execution import context_spec


def accept(root):
    cfg=read(root/'config.json');rows,plan,checks=select_inputs(cfg);probe=check_probe(cfg)
    accept_probe(Path(cfg['jev_probe']))
    spec=context_spec(cfg)
    context=(verify_context(Path(cfg['context_audit'])) if spec is None else
             verify_context(Path(cfg['context_audit']),candidate=spec))
    if (context['container_id']!=cfg['context_container_id']
            or context['result_sha256']!=cfg['context_result_sha256'] or not context['all_inputs_fit']):
        raise ValueError('exact full CPU audit acceptance required')
    launch=read(root/'launch.json');item=inspect(launch['container_id'])
    devices={d['PathOnHost'] for d in item['HostConfig'].get('Devices') or []}
    chips={d for d in devices if d.removeprefix('/dev/davinci').isdigit()}
    if (item['Id']!=launch['container_id'] or item['Image']!=cfg['image'] or launch['image']!=cfg['image']
            or item['State']['Status']!='exited' or item['State']['ExitCode']!=0
            or launch['config_sha256']!=file_hash(root/'config.json')
            or chips not in ({'/dev/davinci4','/dev/davinci5'},{'/dev/davinci6','/dev/davinci7'})):
        raise ValueError('exact successful reserved-NPU shard container required')
    run=root/'run';identity=read(run/'identity.json');result=read(run/'result.json')
    if (read(run/'terminal.json')['exit_code']!=0 or result['status']!='passed'
            or result['weight_updates']!=0 or result['model_files_unchanged'] is not True
            or result['targets_read'] is not False or identity['targets_read'] is not False
            or file_hash(run/'identity.json')!=result['identity_sha256']
            or file_hash(run/'predictions.jsonl')!=result['predictions_sha256']
            or identity['config']!=cfg or identity['versions']!=cfg['versions']
            or identity['model_files']!=probe['model_files'] or identity['inputs_sha256']!=digest(rows)
            or identity['context_result_sha256']!=cfg['context_result_sha256']
            or set(identity['source_files'])!=set(SOURCE_FILES)
            or identity['source_files']!=launch['source_files']):
        raise ValueError('frozen shard result identity changed')
    for name,value in identity['source_files'].items():
        if file_hash(root/'source'/name)!=value:raise ValueError('shard inference source changed')
    predictions=[json.loads(line) for line in (run/'predictions.jsonl').open()]
    check_predictions(rows,checks,predictions)
    keys=[[r['id'],d['id']] for r in rows for d in r['input']['decisions']]
    if (read(run/'selection.json')!=dict(ids=[r['id'] for r in rows],decisions=keys)
            or result['bundles']!=len(rows) or result['decisions']!=len(keys) or result['predictions']!=len(keys)
            or result['length_limited']!=sum(p['length_limited'] for p in predictions)):
        raise ValueError('whole registered shard coverage required')
    return dict(schema='sfc-jev-shard-engine-acceptance-v1',root=str(root.resolve()),container_id=item['Id'],
        shard_plan_sha256=cfg['shard_plan_sha256'],shard_id=cfg['shard_id'],split=cfg['split'],
        prompt_id=cfg['prompt_id'],instruction_sha256=cfg['instruction_sha256'],bundles=len(rows),decisions=len(keys),
        source_groups=len({r['source_group'] for r in rows}),result_sha256=file_hash(run/'result.json'),
        predictions_sha256=file_hash(run/'predictions.jsonl'),verifier_sha256=file_hash(__file__),
        targets_read=False,weight_updates=0,
        scope='Complete single-shard engineering acceptance; no targets or task scores. '
              'Full population requires all registered shards and independent merge/scoring.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=accept(a.root)
    with a.output.open('xb') as handle:handle.write(canonical(result))
    print(json.dumps(result,indent=2))

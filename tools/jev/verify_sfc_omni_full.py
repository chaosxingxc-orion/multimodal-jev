#!/usr/bin/env python3
"""Accept a whole native SFC TEST run before loading targets and official scoring."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_consumption import restore_record
from multimodal_jev.sfc_native_full import select_inputs, check_probe, check_predictions, digest
from infer_sfc_omni_full import SOURCE_FILES
from evaluate_sfc import evaluate
from watch_qwen3_omni_probe import inspect


def accept(root):
    cfg=json.loads((root/'config.json').read_text());rows=select_inputs(cfg);probe=check_probe(cfg)
    launch=json.loads((root/'launch.json').read_text());item=inspect(launch['container_id'])
    devices={d['PathOnHost'] for d in item['HostConfig'].get('Devices') or []}
    npu={d for d in devices if d.removeprefix('/dev/davinci').isdigit()}
    if (item['Id']!=launch['container_id'] or item['Image']!=cfg['image'] or launch['image']!=cfg['image']
            or item['State']['Status']!='exited' or item['State']['ExitCode']!=0
            or launch['config_sha256']!=file_hash(root/'config.json')
            or npu not in ({'/dev/davinci4','/dev/davinci5'},{'/dev/davinci6','/dev/davinci7'})):
        raise ValueError('exact successful reserved-NPU TEST container required')
    predecessor=inspect(cfg['native_probe_container_id'])
    if (predecessor['Id']!=cfg['native_probe_container_id'] or predecessor['Image']!=cfg['image']
            or predecessor['State']['Status']!='exited' or predecessor['State']['ExitCode']!=0):
        raise ValueError('exact successful TRAIN predecessor required')
    run=root/'run';result=json.loads((run/'result.json').read_text());identity=json.loads((run/'identity.json').read_text())
    if (json.loads((run/'terminal.json').read_text())['exit_code']!=0 or result['status']!='passed'
            or result['weight_updates']!=0 or result['model_files_unchanged'] is not True
            or result['targets_read'] is not False or identity['targets_read'] is not False
            or file_hash(run/'identity.json')!=result['identity_sha256']
            or identity['config']!=cfg or identity['versions']!=cfg['versions']
            or identity['model_files']!=probe['model_files'] or identity['inputs_sha256']!=digest(rows)
            or identity['native_probe_acceptance_sha256']!=cfg['native_probe_acceptance_sha256']
            or file_hash(run/'predictions.jsonl')!=result['predictions_sha256']):
        raise ValueError('frozen full TEST identity or predictions changed')
    if (set(identity['source_files'])!=set(SOURCE_FILES)
            or identity['source_files']!=launch['source_files']):
        raise ValueError('complete runner source identity required')
    for name,value in identity['source_files'].items():
        if file_hash(root/'source'/name)!=value:
            raise ValueError('full TEST inference source changed')
    predictions=[json.loads(line) for line in (run/'predictions.jsonl').open()]
    check_predictions(rows,predictions)
    if (json.loads((run/'selection.json').read_text())['ids']!=[r['id'] for r in rows]
            or result['requests']!=len(rows) or result['predictions']!=len(rows)
            or result['length_limited']!=sum(p['length_limited'] for p in predictions)):
        raise ValueError('complete ordered TEST coverage required')
    return cfg,rows,result


def score(root,official_scorer,scorer_hash):
    cfg,rows,result=accept(root)
    # First target read happens after exact-container and complete-output checks.
    release=Path(cfg['release']);manifest=json.loads((release/'MANIFEST.json').read_text())
    target=release/(cfg['split']+'.targets.jsonl')
    if file_hash(target)!=manifest['splits'][cfg['split']]['targets_sha256']:
        raise ValueError('official targets changed')
    targets=[json.loads(line) for line in target.open()]
    if len(targets)!=len(rows):raise ValueError('complete aligned targets required')
    restored=[restore_record(r,t) for r,t in zip(rows,targets,strict=True)]
    path=root/'score-input.jsonl'
    with path.open('xb') as handle:handle.write(b''.join(map(canonical,restored)))
    report=evaluate(path,root/'run/predictions.jsonl',official_scorer,root/'score',scorer_hash)
    receipt=dict(split=cfg['split'],requests=len(rows),source_groups=len({r['source_group'] for r in rows}),
        container_id=json.loads((root/'launch.json').read_text())['container_id'],
        result_sha256=file_hash(root/'run/result.json'),metrics_sha256=file_hash(root/'score/metrics.json'),
        verifier_sha256=file_hash(__file__),weight_updates=0,
        strict_metrics=report['strict_metrics'],strict_conversation_metrics=report['strict_conversation_metrics'],
        scope='Complete official native SFC supplied-history TEST. Strict and official scoring; no interactive execution claim.')
    atomic_json(root/'acceptance.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','official-scorer'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--scorer-hash',required=True);a=p.parse_args()
    print(json.dumps(score(a.root,a.official_scorer,a.scorer_hash),indent=2))

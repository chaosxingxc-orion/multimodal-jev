#!/usr/bin/env python3
"""Validate complete TRAIN engine execution before any full transfer queue."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.omni_slurp_transfer_contract import select_transfer_train,RUNTIME_KEYS
from infer_qwen3_omni_probe import run as inference_run


def verify(root):
    run=root/'run';identity=json.loads((run/'identity.json').read_text());cfg=identity['config']
    result=json.loads((run/'result.json').read_text())
    launch=json.loads((root/'launch.json').read_text());terminal=json.loads((root/'docker-terminal.json').read_text())
    if (json.loads((root/'evaluation-terminal.json').read_text())['exit_code']!=0
            or json.loads((run/'terminal.json').read_text())['exit_code']!=0
            or terminal['container_id']!=launch['container_id']
            or terminal['state']['Status']!='exited' or terminal['state']['ExitCode']!=0
            or launch['physical_devices'] not in ([4,5],[6,7])
            or result['status']!='passed' or result['weight_updates']!=0
            or result['model_files_unchanged'] is not True or result['audio_condition']!='actual'):
        raise ValueError('successful exact TRAIN container and application required')
    if (identity['runner_sha256']!=file_hash(inspect.getfile(inference_run))
            or identity['population_selector_sha256']!=file_hash(inspect.getfile(select_transfer_train))
            or identity['instruction_sha256']!=file_hash(cfg['instruction'])
            or identity['requests_sha256']!=file_hash(cfg['requests'])
            or identity['asset_audit_sha256']!=file_hash(cfg['asset_audit'])
            or result['predictions_sha256']!=file_hash(run/'predictions.jsonl')):
        raise ValueError('TRAIN validation artifacts or active inference source changed')
    source=Path(inspect.getfile(select_transfer_train)).parent
    for name,digest in identity['source_sha256'].items():
        if file_hash(source/name)!=digest:raise ValueError('TRAIN validation source changed')
    rows=select_transfer_train(cfg,[json.loads(line) for line in Path(cfg['requests']).open()])
    ids=json.loads((run/'selection.json').read_text())['ids']
    predictions=[json.loads(line) for line in (run/'predictions.jsonl').open()]
    expected={r['id']:hashlib.sha256(canonical(r)).hexdigest() for r in rows}
    if (ids!=list(expected) or len(predictions)!=336 or result['requests']!=336
            or result['predictions']!=336 or {p['example_id'] for p in predictions}!=set(ids)
            or any(p['request_sha256']!=expected[p['example_id']] or p['split']!='train'
                   or type(p['replicate']) is not int or p['replicate']!=0
                   or p['audio_condition']!='actual' for p in predictions)):
        raise ValueError('complete original TRAIN outputs and bindings required')
    if any(p['input_tokens']+cfg['max_new_tokens']>cfg['max_context_tokens'] for p in predictions):
        raise ValueError('TRAIN request exceeds context budget')
    return dict(schema='jev-omni-transfer-train-validation-v1',root=str(root.resolve()),
        container_id=launch['container_id'],image=launch['image'],source_commit=launch['source_commit'],
        runtime={k:cfg[k] for k in RUNTIME_KEYS},identity_sha256=file_hash(run/'identity.json'),
        result_sha256=file_hash(run/'result.json'),predictions_sha256=file_hash(run/'predictions.jsonl'),
        evaluation_terminal_sha256=file_hash(root/'evaluation-terminal.json'),
        runner_sha256=identity['runner_sha256'],source_sha256=identity['source_sha256'],
        versions=identity['versions'],model_files=identity['model_files'],asset_audit_sha256=identity['asset_audit_sha256'],
        requests=336,weight_updates=0,scope='Successful full TRAIN engine path, no semantic efficacy threshold or TEST feedback')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();receipt=verify(a.root)
    with a.output.open('xb') as handle:handle.write(canonical(receipt))
    print(json.dumps({k:v for k,v in receipt.items() if k!='model_files'},indent=2))

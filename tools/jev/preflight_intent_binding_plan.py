#!/usr/bin/env python3
"""Inspect prepared binding mounts/imports on CPU, without model construction."""
import argparse
import importlib
import json
from pathlib import Path
import sys
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_binding_resume import validate_config,expected_selection


def run(root,kind):
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU only')
    root=Path(root);cfg=json.loads((root/'config-template.json').read_text())
    plan=json.loads((root/'prepared.json').read_text())
    if file_hash(root/'config-template.json')!=plan['config_template_sha256']:
        raise ValueError('prepared configuration changed')
    if file_hash(root/'source-manifest.json')!=plan['source_manifest_sha256']:
        raise ValueError('prepared source manifest changed')
    for path,digest in cfg['pins'].items():
        if file_hash(path)!=digest:raise ValueError('mounted input changed: '+path)
    for path,digest in json.loads((root/'source-manifest.json').read_text()).items():
        if file_hash(root/'source'/path)!=digest:raise ValueError('frozen source changed')
    sys.path.insert(0,str(root/'source/tools/jev'))
    if kind=='training':
        validate_config(cfg);_,selection=expected_selection(cfg)
        module=importlib.import_module('train_intent_binding_npu')
        module.validate_config(cfg)
        if str(root/'source/tools/jev/train_intent_binding_npu.py') not in plan['create_argv']:
            raise ValueError('training command uses wrong entrypoint')
    elif kind=='suite':
        for name in ('verify_intent_binding_npu','run_intent_binding_acceptance'):importlib.import_module(name)
        if cfg['population']!=dict(recordings=89,requests=712):raise ValueError('full binding suite required')
        parent=Path(cfg['training_root']);parent_cfg=json.loads((parent/'config-template.json').read_text())
        validate_config(parent_cfg);_,selection=expected_selection(parent_cfg)
        if cfg['adapter']!=str(parent/'run/adapter') or cfg['adapter_id'] is not None:
            raise ValueError('future updated adapter required')
        if str(root/'source/tools/jev/run_intent_binding_acceptance.py') not in plan['create_argv']:
            raise ValueError('suite command uses wrong entrypoint')
    else:raise ValueError('known prepared phase required')
    return dict(passed=True,kind=kind,pinned_files=len(cfg['pins']),selection=selection,
        source_manifest_sha256=plan['source_manifest_sha256'],config_template_sha256=plan['config_template_sha256'],
        scope='Actual CPU container mount/hash/import and learning-source checks; no model or accepted policy, no NPU dispatch.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--kind',choices=('training','suite'),required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable preflight result exists')
    atomic_json(a.output,run(a.root,a.kind))

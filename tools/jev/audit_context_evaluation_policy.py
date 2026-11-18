#!/usr/bin/env python3
"""Bind a collected evaluation snapshot to its collected training snapshot.

Policy IDs include serialized assets. Never relabel prediction IDs just because
two checkpoints share a step number. This checks weights/contracts/processors;
the selection collector still owns prediction/input/parity authentication.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_policy_artifacts import read_execution_bundle
from context_replay_audit import require_exact


def load_bound(entry):
    proof_path=Path(entry['proof'])
    if file_hash(proof_path)!=entry['proof_sha256']:
        raise ValueError('collection proof changed')
    proof=json.loads(proof_path.read_text())
    state=proof['state']
    if state['Status']!='exited' or state['ExitCode']!=0 or state['OOMKilled']:
        raise ValueError('successful terminal collection required')
    root=Path(entry['collection']).resolve()
    relative=Path(entry['policy_relative_path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('relative policy path required')
    path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('policy outside collection')
    for asset in path.rglob('*'):
        if asset.is_symlink():raise ValueError('policy symlink forbidden')
        if asset.is_file():
            name=str(asset.relative_to(root))
            if proof['files'].get(name)!=file_hash(asset):
                raise ValueError('policy asset not bound by collector: '+name)
    manifest=json.loads((path/'MANIFEST.json').read_text())
    if manifest['policy_id']!=entry['policy_id']:
        raise ValueError('expected checkpoint identity differs')
    bundle=read_execution_bundle(path,expected_policy_id=entry['policy_id'],
        expected_contract=manifest['contract'],expected_base_id=manifest['contract']['base_id'])
    if (bundle['contract']['token']['registration_status']!='admitted' or
            bundle['contract']['request_version']!='jev-context-exclusive_event-v1'):
        raise ValueError('admitted exclusive-event policy required')
    processors={name:digest for name,digest in manifest['files'].items() if name.startswith('processor/')}
    return bundle,processors,proof['container_id']


def run(manifest_path):
    manifest=json.loads(Path(manifest_path).read_text())
    if set(manifest)!={'source','evaluation'}:
        raise ValueError('explicit source and evaluation bindings required')
    source,source_processor,source_container=load_bound(manifest['source'])
    evaluation,evaluation_processor,evaluation_container=load_bound(manifest['evaluation'])
    require_exact(source['contract'],evaluation['contract'],'contract')
    require_exact(source_processor,evaluation_processor,'processor')
    require_exact(source['tensors'],evaluation['tensors'],'execution')
    return dict(source_policy_id=source['policy_id'],inference_policy_id=evaluation['policy_id'],
        same_serialized_policy_id=source['policy_id']==evaluation['policy_id'],
        exact_execution_tensors=len(source['tensors']),processor_exact=True,contract_exact=True,
        source_container=source_container,evaluation_container=evaluation_container,
        bindings=manifest,manifest_sha256=file_hash(manifest_path),
        scope='Collected policy equivalence only. No prediction relabeling, full selection, '
              'continuous training, checkpoint selection or research dispatch admission.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    report=run(args.manifest)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')

"""Export assembled training state without loading a second audio backbone.

The authenticated source owns the frozen marker, processor and contract.
Only complete BF16 LoRA/head execution tensors may be replaced. Optimizer
recovery remains the four-rank checkpoint's job; publication is not admission.
"""
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch

from .baseline import file_hash
from .context_fsdp_export import canonical_execution
from .context_policy_artifacts import read_execution_bundle
from .context_policy_snapshot import _hash
from .local_checkpoint import sync_directory
from .local_runtime_storage import check_paths


def export_checkpoint_policy(source,destination,execution,*,policy_id,contract,base_id):
    source,destination=Path(source),Path(destination)
    bundle=read_execution_bundle(source,expected_policy_id=policy_id,
        expected_contract=contract,expected_base_id=base_id)
    updated=canonical_execution(execution)
    expected={k:v for k,v in bundle['tensors'].items() if k!='decision_token_row'}
    if updated.keys()!=expected.keys() or any(updated[k].shape!=v.shape for k,v in expected.items()):
        raise ValueError('complete compatible assembled execution required')
    check_paths({'joint_policy':destination})
    if destination.exists():raise FileExistsError(destination)
    stage=Path(tempfile.mkdtemp(prefix='.'+destination.name+'-',dir=destination.parent))
    try:
        shutil.copytree(source/'processor',stage/'processor')
        tensors={k:v.detach().clone() for k,v in updated.items()}
        tensors['decision_token_row']=bundle['tensors']['decision_token_row'].clone()
        torch.save(tensors,stage/'execution.pt')
        files={str(p.relative_to(stage)):file_hash(p) for p in stage.rglob('*') if p.is_file()}
        payload=dict(schema='context-policy-execution-v1',contract=contract,files=files)
        identity=_hash(payload)
        (stage/'MANIFEST.json').write_text(json.dumps(dict(payload,policy_id=identity),indent=2)+'\n')
        # Read back every asset before publishing an activation target.
        read_execution_bundle(stage,expected_policy_id=identity,expected_contract=contract,expected_base_id=base_id)
        for p in stage.rglob('*'):
            if p.is_file():
                with p.open('rb') as stream:
                    os.fchmod(stream.fileno(),0o644);os.fsync(stream.fileno())
        for p in sorted((p for p in stage.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
            p.chmod(0o755)
            sync_directory(p)
        stage.chmod(0o755)
        sync_directory(stage);os.rename(stage,destination);sync_directory(destination.parent)
        return dict(policy_id=identity,source_policy_id=policy_id,parameters=len(updated),
            frozen_marker_preserved=True,contract=contract,optimizer_restored=False)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise

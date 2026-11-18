#!/usr/bin/env python3
"""Export a real DP checkpoint as a complete policy without an audio backbone."""
import argparse,json
from pathlib import Path
from multimodal_jev.baseline import file_hash,atomic_json


def run(c,out):
    import torch
    from multimodal_jev.context_checkpoint_policy import export_checkpoint_policy
    from multimodal_jev.context_fsdp_export import canonical_execution
    from multimodal_jev.context_policy_artifacts import read_execution_bundle,export_native_adapter,verify_native_adapter
    from multimodal_jev.local_runtime_storage import require_training_storage
    from multimodal_jev.span_request_identity import tensor_identity
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU only')
    torch.set_num_threads(1);require_training_storage(out)
    for path,digest in c['pins'].items():
        if file_hash(path)!=digest:raise ValueError('pinned dependency changed: '+path)
    tensors=torch.load(c['assembled_execution'],map_location='cpu',weights_only=True)
    canonical=canonical_execution(tensors)
    original=read_execution_bundle(c['source_policy'],expected_policy_id=c['source_policy_id'],
        expected_contract=c['source_contract'],expected_base_id=c['base_id'])
    receipt=export_checkpoint_policy(c['source_policy'],out/'policy',tensors,policy_id=c['source_policy_id'],
        contract=c['source_contract'],base_id=c['base_id'])
    bundle=read_execution_bundle(out/'policy',expected_policy_id=receipt['policy_id'],
        expected_contract=c['source_contract'],expected_base_id=c['base_id'])
    if len(canonical)!=128 or set(bundle['tensors'])!=set(canonical)|{'decision_token_row'}:
        raise ValueError('complete real checkpoint required')
    if any(not torch.equal(bundle['tensors'][k],v) for k,v in canonical.items()):
        raise ValueError('execution changed during export')
    if not torch.equal(bundle['tensors']['decision_token_row'],original['tensors']['decision_token_row']):
        raise ValueError('frozen marker changed')
    export_native_adapter(out/'adapter',bundle);verify_native_adapter(out/'adapter',bundle)
    atomic_json(out/'result.json',dict(passed=True,receipt=receipt,
        tensors={k:tensor_identity(v) for k,v in bundle['tensors'].items()},
        scope='Actual DP execution exported and independently compared on CPU, derived native adapter verified; no engine activation or task gain.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False);run(json.loads(a.config.read_text()),a.output)

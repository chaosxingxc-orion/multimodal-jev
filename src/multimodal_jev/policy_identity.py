"""Hash model and processor assets consistently for rollout and actor gates."""
import hashlib
import json
from pathlib import Path
from .baseline import file_hash


def checkpoint_identity(checkpoint):
    root=Path(checkpoint)
    files={str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*'))
           if p.is_file() and p.suffix in {'.json','.jinja','.safetensors','.model','.txt'}}
    if not any(name.endswith('.safetensors') for name in files):
        raise ValueError('model weights missing')
    return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),files


def compare_adapter_artifacts(reference, candidate):
    """Compare exact weights and PEFT config, allowing target-module set order only.

    Preserve both original artifact identities. This is a comparison receipt,
    never a replacement policy ID or permission to skip probability/state gates.
    """
    roots = [Path(reference), Path(candidate)]
    identities = [checkpoint_identity(p) for p in roots]
    files = [item[1] for item in identities]
    required = {'adapter_config.json', 'adapter_model.safetensors'}
    if not required <= files[0].keys() or files[0].keys() != files[1].keys():
        raise ValueError('same complete adapter file set required')
    for name in files[0]:
        if name != 'adapter_config.json' and files[0][name] != files[1][name]:
            raise ValueError('exact adapter weight/asset bytes differ: '+name)
    configs = [json.loads((p/'adapter_config.json').read_text()) for p in roots]
    for config in configs:
        targets = config.get('target_modules')
        if (not isinstance(targets, list) or not targets
                or any(not isinstance(t, str) or not t for t in targets)
                or len(set(targets)) != len(targets)):
            raise ValueError('explicit unique target module list required')
        config['target_modules'] = sorted(targets)
    if configs[0] != configs[1]:
        raise ValueError('adapter configuration semantics differ')
    return dict(passed=True, reference_id=identities[0][0], candidate_id=identities[1][0],
                reference_files=files[0], candidate_files=files[1], exact_weight_bytes=True,
                config_equivalence='target_modules_set_order_only')


def validate_adapter_export(base, checkpoint, adapter, *, expected_training_run=None,
                            base_files=None, checkpoint_files=None):
    """Verify persisted base/adapter/export lineage without loading a model.

    A passed EXPORT receipt is insufficient if its published weights have since
    changed. Check its file hashes as well as the adapter and original base.
    This verifies artifact provenance, not actor/rollout numerical parity.
    """
    base,checkpoint,adapter=map(Path,(base,checkpoint,adapter))
    base_files=checkpoint_identity(base)[1] if base_files is None else base_files
    checkpoint_files=(checkpoint_identity(checkpoint)[1]
                      if checkpoint_files is None else checkpoint_files)
    files={p.name:file_hash(p) for p in sorted(adapter.iterdir()) if p.is_file()}
    if not {'adapter_config.json','adapter_model.safetensors'}<=files.keys():
        raise ValueError('adapter configuration and weights are required')
    receipt=json.loads((checkpoint/'EXPORT.json').read_text())
    if (receipt.get('status')!='passed' or receipt.get('adapter_files')!=files
            or receipt.get('base_sha256')!=base_files.get('model.safetensors')):
        raise ValueError('adapter/base does not match rollout export')
    exported=receipt.get('export_files',{})
    if not {'config.json','model.safetensors'}<=exported.keys():
        raise ValueError('export receipt lacks model file hashes')
    actual={str(p.relative_to(checkpoint)) for p in checkpoint.rglob('*')
            if p.is_file() and p!=checkpoint/'EXPORT.json'}
    if actual!=set(exported):
        raise ValueError('published export file set changed')
    for name,digest in exported.items():
        observed=checkpoint_files.get(name)
        if observed is None:observed=file_hash(checkpoint/name)
        if observed!=digest:raise ValueError(f'published export file changed: {name}')
    if expected_training_run is not None:
        if not receipt.get('training_run') or Path(receipt['training_run']).resolve()!=Path(expected_training_run).resolve():
            raise ValueError('rollout is not an export of the expected training run')
    return files

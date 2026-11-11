"""Atomic, authenticated BF16 adapter + master optimizer checkpoints."""
import json
import os
from pathlib import Path
import torch

from .baseline import atomic_json, file_hash
from .policy_identity import checkpoint_identity


def save_span_checkpoint(destination, adapted, model, optimizer, base_id, updates):
    destination = Path(destination)
    pending = destination.with_name(destination.name + '.pending')
    if destination.exists() or pending.exists():
        raise ValueError('checkpoint version already exists or has an incomplete transaction')
    pending.mkdir(parents=True)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if (len(named) != len(optimizer.parameters)
            or any(p is not q for (_, p), q in zip(named, optimizer.parameters))):
        raise ValueError('optimizer parameter order differs from named execution model')
    if any(p.dtype != torch.bfloat16 for _, p in named):
        raise ValueError('BF16 execution checkpoint required')
    adapted.save_pretrained(pending / 'adapter')
    adapter_id, adapter_files = checkpoint_identity(pending / 'adapter')
    torch.save(dict(optimizer=optimizer.state_dict(), ordered_parameter_names=[n for n, _ in named],
                    base_id=base_id, adapter_id=adapter_id, completed_updates=updates), pending / 'optimizer.pt')
    atomic_json(pending / 'CHECKPOINT.json', dict(schema='jev-span-checkpoint-v1', base_id=base_id,
        adapter_id=adapter_id, adapter_files=adapter_files, completed_updates=updates,
        optimizer_sha256=file_hash(pending / 'optimizer.pt')))
    os.rename(pending, destination)
    return base_id + ':' + adapter_id


def restore_span_optimizer(checkpoint, model, optimizer, base_id):
    """Execution adapter must already be loaded. Verify before restoring state."""
    checkpoint = Path(checkpoint)
    receipt = json.loads((checkpoint / 'CHECKPOINT.json').read_text())
    adapter_id, files = checkpoint_identity(checkpoint / 'adapter')
    if (receipt.get('schema') != 'jev-span-checkpoint-v1' or receipt['base_id'] != base_id
            or receipt['adapter_id'] != adapter_id or receipt['adapter_files'] != files
            or receipt['optimizer_sha256'] != file_hash(checkpoint / 'optimizer.pt')):
        raise ValueError('checkpoint identity mismatch')
    state = torch.load(checkpoint / 'optimizer.pt', map_location='cpu', weights_only=True)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    params = [p for p in model.parameters() if p.requires_grad]
    if (state['ordered_parameter_names'] != names or state['base_id'] != base_id
            or state['adapter_id'] != adapter_id or state['completed_updates'] != receipt['completed_updates']
            or len(params) != len(optimizer.parameters)
            or any(p is not q for p, q in zip(params, optimizer.parameters))):
        raise ValueError('checkpoint parameter order or lineage mismatch')
    optimizer.load_state_dict(state['optimizer'])
    return receipt

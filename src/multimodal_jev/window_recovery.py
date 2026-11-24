"""Local, fixed-topology recovery for a single admitted formal training window."""
import json
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

from .baseline import file_hash
from .local_checkpoint import commit_file, commit_json, sync_directory


def restart_enabled(config):
    return bool(config and int(os.environ.get('TORCHELASTIC_RESTART_COUNT', '0')) > 0)


def latest(root, contract, first, end, world):
    directory = root/'recovery'
    if not directory.exists():
        return None
    candidates = sorted(directory.glob('step-*.json'), key=lambda p: int(p.stem[5:]))
    if not candidates:
        return None
    path = candidates[-1]
    proof = json.loads(path.read_bytes())
    step = proof['step']
    if (type(step) is not int or not first < step < end or path.stem != f'step-{step}'
            or proof['contract'] != contract or len(proof['shards']) != world
            or not re.fullmatch(r'generation-[0-9a-f]{32}',proof['generation'])):
        raise ValueError('recovery cursor, topology or window differs')
    return proof


def save(root, named, optimizer, identity, auxiliary, dist):
    import torch
    from .context_training_checkpoint import capture
    rank, world, step = (identity[k] for k in ('rank','world_size','completed_updates'))
    directory = root/'recovery'
    group = ['generation-'+uuid4().hex if rank == 0 else None]
    dist.broadcast_object_list(group, src=0)
    generation = directory/group[0]
    generation.mkdir(parents=True, exist_ok=True)
    path = generation/f'rank-{rank}.pt'
    state = capture(named, optimizer, **identity, auxiliary=auxiliary)
    commit_file(path, lambda stream: torch.save(state, stream))
    hashes = [None]*world
    dist.all_gather_object(hashes, file_hash(path))
    if rank == 0:
        # The new generation is durable before the prior generation can be retired.
        sync_directory(directory)
        commit_json(directory/f'step-{step}.json', dict(step=step, contract=identity['contract'],
                    generation=group[0], shards=hashes))
        for prior in directory.glob('step-*.json'):
            if prior.stem != f'step-{step}':
                prior.unlink()
        sync_directory(directory)
        for prior in directory.glob('generation-*'):
            if prior != generation and re.fullmatch(r'generation-[0-9a-f]{32}',prior.name):
                shutil.rmtree(prior)
    dist.barrier()


def recover(root, proof, named, optimizer, identity):
    import torch
    from .context_training_checkpoint import restore
    path = root/'recovery'/proof['generation']/f"rank-{identity['rank']}.pt"
    if file_hash(path) != proof['shards'][identity['rank']]:
        raise ValueError('recovery shard digest differs')
    state = torch.load(path, map_location='cpu', weights_only=True)
    restore(named, optimizer, state, **identity)
    return state.get('auxiliary', {})


def preserve_tail(root, paths):
    """Only move explicitly named, uncommitted outputs in this formal window."""
    paths = [p for p in paths if p.exists()]
    if paths:
        quarantine = root/('interrupted-'+uuid4().hex)
        quarantine.mkdir()
        for path in paths:
            path.rename(quarantine/path.name)
        sync_directory(root)

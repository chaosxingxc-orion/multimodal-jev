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


def fixed_epoch_parent(config, world):
    """Authenticate the interrupted formal window and its durable cursor."""
    from .slurp_training import content_hash

    pin = config['fixed_epoch_recovery']
    formal = Path(pin['root'])
    parent_path = formal/'config.json'
    if file_hash(parent_path) != pin['formal_sha256']:
        raise ValueError('interrupted formal configuration changed')
    parent = json.loads(parent_path.read_bytes())
    allowed = {'fixed_epoch_recovery', 'start_update', 'finite_graph_admission'}
    if config['engineering_prefix']:
        allowed.update({'engineering_prefix', 'training_dispatch_allowed', 'stop_update',
                        'checkpoint_interval', 'parity_passes', 'recovery', 'evaluation_steps'})
    if (parent['engineering_prefix'] is not False or parent['training_dispatch_allowed'] is not True
            or 'instance_negative_manifest' not in parent or 'token_buckets' not in parent
            or world != len(parent['physical_devices'])
            or config['start_update'] != pin['step']
            or {k: v for k, v in config.items() if k not in allowed}
            != {k: v for k, v in parent.items() if k not in allowed}):
        raise ValueError('same-epoch recovery changes the frozen training configuration')
    root = formal/'run'
    cursor = root/'recovery'/f"step-{pin['step']}.json"
    if file_hash(cursor) != pin['cursor_sha256']:
        raise ValueError('interrupted recovery cursor changed')
    end = parent['batch_config']['source_queries'] * parent['batch_config']['epochs']
    contract = dict(stage='stage1', config=content_hash(parent),
                    start=parent['start_update'], stop=end)
    proof = latest(root, contract, parent['start_update'], end, world)
    if proof is None or proof['step'] != pin['step']:
        raise ValueError('recovery must use the latest committed cursor')
    return root, proof, contract


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

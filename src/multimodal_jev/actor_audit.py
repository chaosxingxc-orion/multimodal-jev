"""Read-only validation of saved fixed-base LoRA/AdamW actor artifacts."""
import json
import math
from pathlib import Path

from .baseline import file_hash


def adapter_name(name):
    # PEFT omits the active adapter name when saving its safetensors file.
    return name.replace('.lora_A.default.', '.lora_A.').replace('.lora_B.default.', '.lora_B.')


def validate_state(state, weights, updates, optimizer_config):
    import torch
    if type(updates) is not int or updates < 0:
        raise ValueError('nonnegative integer update count required')
    names, shapes = state['parameter_names'], state['parameter_shapes']
    keys = [adapter_name(name) for name in names]
    if not names or len(set(keys)) != len(keys) or set(keys) != set(weights) or len(shapes) != len(names):
        raise ValueError('optimizer parameter mapping does not match adapter')
    optimizer = state['optimizer']
    groups = optimizer['param_groups']
    if len(groups) != 1:
        raise ValueError('registered actor requires one AdamW parameter group')
    group = groups[0]
    for key, value in optimizer_config.items():
        if group.get(key) != value:
            raise ValueError('saved optimizer configuration changed: ' + key)
    ids = group['params']
    if len(ids) != len(names) or len(set(ids)) != len(ids):
        raise ValueError('optimizer parameter IDs are incomplete or duplicated')
    moments = optimizer['state']
    if set(moments) != (set(ids) if updates else set()):
        raise ValueError('AdamW state coverage disagrees with updates')
    for key, shape, pid in zip(keys, shapes, ids):
        weight = weights[key]
        if list(weight.shape) != shape or not torch.isfinite(weight).all():
            raise ValueError('adapter shape or finite-value check failed')
        if updates:
            moment = moments[pid]
            step = moment['step']
            if not torch.is_tensor(step) or step.numel() != 1 or not torch.isfinite(step).all() or step.item() != updates:
                raise ValueError('AdamW step differs from recorded update count')
            for field in ('exp_avg', 'exp_avg_sq'):
                value = moment[field]
                if list(value.shape) != shape or not torch.isfinite(value).all():
                    raise ValueError('AdamW moment shape or finite-value check failed')
            if (moment['exp_avg_sq'] < 0).any():
                raise ValueError('negative AdamW second moment')
    rng = state['cpu_rng']
    if not torch.is_tensor(rng) or rng.dtype != torch.uint8 or rng.ndim != 1 or rng.numel() == 0:
        raise ValueError('invalid persisted CPU RNG state')
    return {'adapter_tensors': len(weights), 'optimizer_states': len(moments),
            'optimizer_step': updates, 'all_weights_and_moments_finite': True}


def completed_artifacts(run):
    run = Path(run)
    terminal = json.loads((run/'terminal.json').read_text())
    result = json.loads((run/'result.json').read_text())
    if terminal.get('exit_code') != 0 or result.get('status') != 'passed':
        raise ValueError('successful application terminal and result required')
    files = {p.name: file_hash(p) for p in sorted((run/'adapter').iterdir()) if p.is_file()}
    if files != result['adapter_files'] or file_hash(run/'optimizer.pt') != result['optimizer_sha256']:
        raise ValueError('saved actor artifact hashes changed')
    return result


def audit_actor(run):
    import torch
    from safetensors.torch import load_file
    run = Path(run).resolve()
    result = completed_artifacts(run)
    weights = load_file(str(run/'adapter/adapter_model.safetensors'), device='cpu')
    state = torch.load(run/'optimizer.pt', map_location='cpu', weights_only=True)
    counts = validate_state(state, weights, result['updates'], result['optimizer_config'])
    config, metrics = result['config'], result['metrics']
    updated = metrics['updated']
    if type(updated) is not bool:
        raise ValueError('boolean update outcome required')
    error, limit = metrics['max_logprob_error'], config['max_logprob_error']
    if not math.isfinite(error) or not math.isfinite(limit) or not 0 <= error <= limit:
        raise ValueError('actor/rollout parity gate failed')
    parent = config.get('parent_run')
    previous_updates = 0
    previous_iteration = 0
    reference = config.get('initial_adapter')
    if parent:
        parent = Path(parent).resolve()
        if parent == run or reference:
            raise ValueError('invalid parent/initializer lineage')
        prior = completed_artifacts(parent)
        for key in ('base_policy_id', 'optimizer_config', 'reward_mode', 'pair_bonus_weight', 'reference_policy'):
            if prior.get(key) != result.get(key):
                raise ValueError('actor continuation changed: ' + key)
        from .advantage_config import validate_continuation
        validate_continuation(prior['config'], config)
        previous_updates, previous_iteration = prior['updates'], prior['iteration']
        reference = parent/'adapter'
        prior_weights = load_file(str(reference/'adapter_model.safetensors'), device='cpu')
        prior_state = torch.load(parent/'optimizer.pt', map_location='cpu', weights_only=True)
        validate_state(prior_state, prior_weights, previous_updates, prior['optimizer_config'])
        if state['parameter_names'] != prior_state['parameter_names'] or state['parameter_shapes'] != prior_state['parameter_shapes']:
            raise ValueError('optimizer coordinates changed across continuation')
        if not updated:
            if state['optimizer']['param_groups'] != prior_state['optimizer']['param_groups']:
                raise ValueError('no-op changed optimizer parameter groups')
            for pid, moment in state['optimizer']['state'].items():
                for key, value in moment.items():
                    if not torch.equal(value, prior_state['optimizer']['state'][pid][key]):
                        raise ValueError('no-op changed optimizer moments')
    if result['updates'] != previous_updates + int(updated) or result['iteration'] != previous_iteration + 1:
        raise ValueError('actor iteration/update continuation mismatch')
    changed = None
    if reference:
        reference = Path(reference)
        identity = json.loads((run/'identity.json').read_text())
        files = {p.name: file_hash(p) for p in sorted(reference.iterdir()) if p.is_file()}
        if files != identity['adapter_files']:
            raise ValueError('input adapter differs from actor identity')
        before = load_file(str(reference/'adapter_model.safetensors'), device='cpu')
        if set(before) != set(weights) or any(before[k].shape != weights[k].shape for k in weights):
            raise ValueError('input/output adapter tensor coordinates differ')
        if any(not torch.isfinite(value).all() for value in before.values()):
            raise ValueError('nonfinite input adapter weights')
        changed = sum(not torch.equal(weights[key], before[key]) for key in weights)
        if not updated and changed:
            raise ValueError('no-op changed adapter weights')
        if updated and not changed:
            raise ValueError('claimed update changed no adapter weights')
    return {'status': 'passed', 'run': str(run), **counts, 'iteration': result['iteration'],
            'changed_adapter_tensors': changed, 'updated': updated,
            'max_logprob_error': error, 'parent_run': str(parent) if parent else None,
            'result_sha256': file_hash(run/'result.json'),
            'optimizer_sha256': result['optimizer_sha256'], 'adapter_files': result['adapter_files'],
            'scope': 'saved artifact hashes, finite tensors, AdamW counters and local parent consistency; '
                     'not Docker exit, rollout/export integrity, gradient equivalence or efficacy proof'}

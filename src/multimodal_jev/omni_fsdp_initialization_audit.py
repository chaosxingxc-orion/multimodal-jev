"""Validate both rank manifests against accepted CPU structure and wrap layout.

This checks saved evidence, not live tensor values. Runtime/source authentication
and the original loader's checkpoint comparisons remain separate requirements.
"""
import math

from .sfc_reward_batch import valid_hash


def validate_ranks(original, structure, buffers, units, ranks, *, original_sha256,
                   preflight_sha256):
    if (not original or set(original) != set(structure) or len(ranks) != 2
            or not valid_hash(original_sha256) or not valid_hash(preflight_sha256)):
        raise ValueError('complete original structure and two ranks required')
    for name, item in original.items():
        ref = structure[name]
        if (set(item) != {'shape', 'dtype', 'sha256'} or item['shape'] != ref['shape']
                or item['dtype'] != ref['dtype'] or item['dtype'] != 'torch.bfloat16'
                or not valid_hash(item['sha256']) or math.prod(item['shape']) != ref['numel']):
            raise ValueError('original parameter identity differs from accepted CPU structure')
    expected_units = {u['name']: u['parameters'] for u in units}
    total = sum(v['numel'] for v in structure.values())
    if len(expected_units) != len(units) or sum(expected_units.values()) != total:
        raise ValueError('complete unique wrap layout required')
    # Derive ownership independently from longest enclosing module prefix.
    counts = dict.fromkeys(expected_units, 0)
    for name, item in structure.items():
        owners = [u for u in counts if u != '<root>' and name.startswith(u+'.')]
        owner = max(owners, key=len) if owners else '<root>'
        if owner not in counts:
            raise ValueError('root wrap unit missing')
        counts[owner] += item['numel']
    if counts != expected_units:
        raise ValueError('wrap ownership differs from complete parameter structure')
    expected_buffers = {n: {k: v for k, v in b.items() if k != 'persistent'}
                        for n, b in buffers.items()}
    seen = set(); memory = []
    for result, parameters in ranks:
        rank = result['rank']
        if (type(rank) is not int or rank not in (0, 1) or rank in seen
                or type(result['local_rank']) is not int or result['local_rank'] != rank
                or result['device'] != f'npu:{rank}' or result['visible_devices'] != 2
                or result['status'] != 'initialization_verified'
                or result['parameters'] != total or parameters != original
                or result['original_parameters_sha256'] != original_sha256
                or result['preflight_sha256'] != preflight_sha256
                or result['buffers'] != expected_buffers
                or result['forward_calls'] != 0 or result['weight_updates'] != 0):
            raise ValueError('complete matching frozen rank identity and state required')
        seen.add(rank)
        observed = {u['name']: u['parameters'] for u in result['units']}
        if len(observed) != len(result['units']) or observed != expected_units:
            raise ValueError('rank wrap units incomplete or changed')
        initial = result['initial_memory']
        values = [initial['allocated'], initial['reserved'],
                  result['peak_allocated_bytes'], result['peak_reserved_bytes']]
        if (any(type(v) is not int or v <= 0 for v in values)
                or values[0] > values[1] or values[0] > values[2]
                or values[1] > values[3] or values[2] > values[3]
                or type(result['elapsed_seconds']) not in (int, float)
                or not math.isfinite(result['elapsed_seconds']) or result['elapsed_seconds'] < 0):
            raise ValueError('valid initialization allocator accounting required')
        memory.append(dict(rank=rank, initial=initial, peak_allocated_bytes=values[2],
                           peak_reserved_bytes=values[3], elapsed_seconds=result['elapsed_seconds']))
    return dict(parameters=total, parameter_tensors=len(original), buffers=len(buffers),
                units=len(units), ranks=2, memory=sorted(memory, key=lambda r: r['rank']))

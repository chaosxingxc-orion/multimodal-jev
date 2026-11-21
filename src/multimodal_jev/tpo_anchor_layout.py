"""Separate offline fixed-reference workers from training-time TP1 services."""
import json
from pathlib import Path


def device_plan(training, online, offline):
    for scope in (training, online, offline):
        if (not isinstance(scope, list) or not scope
                or any(type(card) is not int or card < 0 for card in scope)
                or len(set(scope)) != len(scope)):
            raise ValueError('TPO device lists must contain distinct nonnegative card IDs')
    if set(training) & set(online) or set(offline) != set(training) | set(online):
        raise ValueError('Offline fixed reference must use every card in the training and TP1 allocation')
    return dict(fixed_reference=offline, actor=training, online_reference=online)


def window_device_plan(config):
    if full := config.get('full_joint'):
        training, online = full['training_devices'], full['inference_devices']
    else:
        training = json.loads(Path(config['parent_config']).read_bytes())['physical_devices']
        online = config['joint']['inference_devices']
    return device_plan(training, online, config['offline_reference_devices'])


def partition(world, updates, workers):
    """Split equal-sized rank-step work while keeping each inference microbatch intact."""
    count = world * updates
    if min(world, updates, workers) < 1 or workers > count:
        raise ValueError('Every fixed-reference worker must have nonempty work')
    size, extra = divmod(count, workers)
    begin = 0
    result = []
    for i in range(workers):
        end = begin + size + (i < extra)
        parts = []
        while begin < end:
            rank, start = divmod(begin, updates)
            stop = min(end - rank * updates, updates)
            parts.append((rank, start, stop))
            begin = rank * updates + stop
        result.append(parts)
    return result


def part_name(rank, start, stop):
    return f'rank-{rank}-steps-{start}-{stop}.pt'


def checked_parts(receipts, plan, joint_sha, updates, rank=None):
    """Validate the entire roster and return the ordered parts for one rank."""
    cards = device_plan(plan['training_devices'], plan['inference_devices'],
                        plan['offline_reference_devices'])['fixed_reference']
    assignments = partition(len(plan['training_devices']), updates, len(cards))
    selected = []
    for index, (card, parts, receipt) in enumerate(zip(cards, assignments, receipts, strict=True)):
        files = receipt['parts']
        if (receipt['card'] != card or receipt['source_policy'] != plan['source_policy']
                or receipt['target_sha'] != plan['target_sha'] or receipt['joint_sha'] != joint_sha
                or receipt['updates'] != updates or set(files) != {part_name(*p) for p in parts}):
            raise ValueError('Unbound or incomplete fixed-reference shard')
        selected.extend((index, part_name(r, start, stop), files[part_name(r, start, stop)], stop-start)
                        for r, start, stop in parts if r == rank)
    return selected

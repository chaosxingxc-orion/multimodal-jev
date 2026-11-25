"""Bind physical NPU allocations to the knowledge-card experiment schedules."""


def physical_devices(value):
    if (not isinstance(value, list) or not value
            or any(type(i) is not int or i < 0 for i in value)
            or len(set(value)) != len(value)):
        raise ValueError('distinct nonnegative physical device IDs required')
    return value


def device_count(pool, config, key):
    count = len(physical_devices(pool))
    if key in config and (type(config[key]) is not int or config[key] != count):
        raise ValueError(f'{key} differs from the physical device count')
    return count


def tpo_schedule(layout, schedule):
    if not isinstance(layout, dict) or set(layout) != {'training_devices', 'per_dp_replica_batch'}:
        raise ValueError('explicit card TPO training devices and effective per-rank batch required')
    pool = physical_devices(layout['training_devices'])
    local, micro = layout['per_dp_replica_batch'], schedule['microbatch']
    if (type(local) is not int or type(micro) is not int or micro < 1
            or local < 4 * micro or local % micro):
        raise ValueError('card TPO needs four complete parity microbatches per rank')
    global_batch = local * len(pool)
    if 'global_batch' in schedule and (type(schedule['global_batch']) is not int
                                       or schedule['global_batch'] != global_batch):
        raise ValueError('card TPO schedule differs from configured training devices')
    return dict(schedule, global_batch=global_batch)

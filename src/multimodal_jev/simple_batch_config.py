"""Explicit DP4 batch candidates; changing batch size requires a new schedule."""


def batch_layout(config):
    global_batch = config['global_batch']
    micro = config.get('microbatch_per_rank', 1)
    if (type(global_batch) is not int or global_batch not in (128, 256)
            or type(micro) is not int or micro not in (1, 2, 4)
            or global_batch % (4 * micro)):
        raise ValueError('DP4 global128/256 and dividing micro1/2/4 required')
    return dict(global_batch=global_batch, world_size=4, local_batch=global_batch//4,
                microbatch_per_rank=micro, accumulation_per_rank=global_batch//(4*micro))

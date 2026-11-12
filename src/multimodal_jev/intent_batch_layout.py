"""Explicit equal-row DP4 layouts for separately registered coverage controls."""


def local_requests(config):
    count=config['global_requests']
    if type(count) is not int or count not in (8,32):
        raise ValueError('registered global batch8 or batch32 required')
    return count//4


def rank_slice(config, rank):
    if type(rank) is not int or not 0<=rank<4:raise ValueError('physical DP4 rank required')
    size=local_requests(config)
    return slice(rank*size,(rank+1)*size)

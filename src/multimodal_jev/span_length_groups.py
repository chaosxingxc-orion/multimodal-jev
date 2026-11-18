"""Deterministic permutations of a fixed optimizer batch, without reading labels.

Fixed counts retain the existing native verl loss scaling. This is a candidate
planner, not dynamic-bsz admission or an epoch sampler.
"""


def length_group_indices(lengths, *, microbatch, world_size, order):
    if (order not in ('original', 'length_grouped')
            or type(microbatch) is not int or microbatch < 1
            or type(world_size) is not int or world_size < 1
            or not lengths or len(lengths) % (world_size * microbatch)
            or any(len(row) != 2 or any(type(v) is not int or v < 1 for v in row)
                   for row in lengths)):
        raise ValueError('positive (tokens, frames), complete fixed micro/DP batch required')
    local = len(lengths) // world_size
    if order == 'original':
        return [list(range(r * local, (r + 1) * local)) for r in range(world_size)]
    indices = sorted(range(len(lengths)), key=lambda i: (*lengths[i], i))
    ranks = [[] for _ in range(world_size)]
    # Alternate which rank receives each consecutive length group. This limits
    # persistent long-rank assignment across rounds; actual runtime still matters.
    for offset in range(0, len(indices), world_size * microbatch):
        round_index = offset // (world_size * microbatch)
        for rank in range(world_size):
            target = rank if round_index % 2 == 0 else world_size - 1 - rank
            start = offset + rank * microbatch
            ranks[target].extend(indices[start:start + microbatch])
    if sorted(i for rank in ranks for i in rank) != list(range(len(lengths))):
        raise ValueError('planner lost or duplicated slots')
    return ranks

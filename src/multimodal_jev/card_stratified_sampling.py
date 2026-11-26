"""Select frozen TRAIN recording indices with source diversity and directed strata."""
from collections import defaultdict, deque
import random


def sample(rows, high_counts, policy):
    n, replay = policy['budget'], policy['replay_recordings']
    if len(high_counts) != len(rows) or not 0 <= replay <= n <= len(rows):
        raise ValueError('invalid sampling population or budget')
    indices = list(range(len(rows)))
    random.Random(policy['seed']).shuffle(indices)
    fixed = set(policy['retain_all'])
    if fixed & {'replay', 'boundary'} or any(r['stratum'] not in fixed | {'replay', 'boundary'} for r in rows):
        raise ValueError('unregistered or overlapping stratum')
    chosen = [i for i in indices if rows[i]['stratum'] in fixed]
    easy = [i for i in indices if rows[i]['stratum'] == 'replay']
    if len(easy) < replay or len(chosen) + replay > n:
        raise ValueError('cannot preserve mandatory strata and replay at requested budget')
    chosen.extend(easy[:replay])
    groups = {rows[i]['group'] for i in chosen}
    pools = defaultdict(deque)
    for i in indices:
        r = rows[i]
        if r['stratum'] == 'boundary':
            key = (r['gold'], r['competitor'], high_counts[i] >= policy['many_rivals'], r['synthetic'])
            pools[key].append(i)
    active, deferred = deque(pools.values()), deque()
    while active and len(chosen) < n:
        pool = active.popleft()
        skipped = deque()
        while pool and rows[pool[0]]['group'] in groups:
            skipped.append(pool.popleft())
        if skipped:
            deferred.append(skipped)
        if pool:
            i = pool.popleft()
            chosen.append(i)
            groups.add(rows[i]['group'])
        if pool:
            active.append(pool)
    while len(chosen) < n:
        if not deferred:
            raise ValueError('not enough eligible boundary recordings')
        pool = deferred.popleft()
        chosen.append(pool.popleft())
        if pool:
            deferred.append(pool)
    return sorted(chosen)

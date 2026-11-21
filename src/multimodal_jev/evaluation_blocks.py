"""Schedule frozen evaluation batches across a pool of interchangeable workers."""
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import re

from .context_execution import content_key
from .local_checkpoint import commit_json
from .local_runtime_storage import check_paths

SCHEMA = 'jev-evaluation-blocks-v1'


def make_blocks(dataset, roster, arms, *, batch, batches_per_block, partitions, choice):
    """Keep the legacy batch boundaries, including each logical partition's tail."""
    if any(type(n) is not int or n < 1 for n in (batch, batches_per_block, partitions)):
        raise ValueError('positive evaluation batch, block size and logical partitions required')
    if not arms or not roster or partitions > len(roster):
        raise ValueError('nonempty evaluation arms, roster and partitions required')
    size = batch * batches_per_block
    for index in range(partitions):
        shard = roster[index::partitions]
        for start in range(0, len(shard), size):
            rows = shard[start:start + size]
            # Estimate padded work from the already audited encoded lengths.
            # A full ontology Choice pass adds one marker and one forward.
            work = 0
            for offset in range(0, len(rows), batch):
                group = rows[offset:offset + batch]
                padded = max(r['tokens'] for r in group) + max(r['frames'] for r in group)
                work += len(group) * (padded if not choice else 2 * padded + 1)
            for arm in arms:
                yield dict(dataset=dataset, arm=arm, partition=index, offset=start,
                           ids=[r['id'] for r in rows], widths=[r['width'] for r in rows], work=work)


def ordered_blocks(blocks, arms):
    # Run one policy across datasets before moving to the next. Workers advance
    # as soon as all earlier work is claimed; they need not wait at a barrier.
    rank = {arm: index for index, arm in enumerate(arms)}
    rows = sorted(blocks, key=lambda b: (rank[b['arm']], -b['work'], b['dataset'], b['partition'], b['offset']))
    return [dict(block, key=f'block-{index:06d}') for index, block in enumerate(rows)]


def read(path):
    return json.loads(path.read_bytes())


def checked_plan(root, namespace):
    manifest = read(Path(root) / 'plan.json')
    plan = manifest['plan']
    if (plan['schema'] != SCHEMA or plan['namespace'] != namespace
            or manifest['sha256'] != content_key(plan)):
        raise ValueError('evaluation block plan changed')
    return manifest


def checked_payload(payload, manifest, block):
    plan = manifest['plan']
    keys = {'prob': 'prob', **({'choices': 'logp'} if plan['choice'] else {})}
    products = payload['products']
    if (payload['plan_sha256'] != manifest['sha256'] or payload['block'] != block['key']
            or set(products) != set(keys) or payload['sha256'] != content_key(products)):
        raise ValueError('evaluation block identity or content differs: ' + block['key'])
    for kind, field in keys.items():
        result = products[kind]
        rows = result['records']
        if (result['policy_id'] != plan['policies'][block['arm']]
                or [row['id'] for row in rows] != block['ids']):
            raise ValueError('evaluation block policy or population differs: ' + block['key'])
        for row, expected in zip(rows, block['widths'], strict=True):
            width = expected + (kind == 'choices')
            values = row[field]
            if (len(values) != width or any(
                    type(v) not in (int, float) or not math.isfinite(v)
                    or (kind == 'prob' and not 0 <= v <= 1) or (kind == 'choices' and v > 0) for v in values)):
                raise ValueError('invalid evaluation block scores: ' + block['key'])
    return products


def artifact(root, block):
    return Path(root) / (block['key'] + '.json')


def read_block(root, manifest, block):
    return checked_payload(read(artifact(root, block)), manifest, block)


def publish(root, manifest, block, products):
    payload = dict(plan_sha256=manifest['sha256'], block=block['key'],
                   products=products, sha256=content_key(products))
    checked_payload(payload, manifest, block)
    commit_json(artifact(root, block), payload)


@contextmanager
def claim(root, block):
    """A process exit releases its unfinished block; committed files are immutable."""
    key = block['key']
    if re.fullmatch(r'block-\d{6,}', key) is None:
        raise ValueError('invalid evaluation block filename')
    path = artifact(root, block)
    if path.is_file():
        yield False
        return
    check_paths({'evaluation block lock': path})
    with path.with_suffix('.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            yield not path.is_file()

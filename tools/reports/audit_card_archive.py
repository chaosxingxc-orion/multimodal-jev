"""Compare every candidate value in the declared set of frozen blocks."""
from functools import partial
from pathlib import Path

from multimodal_jev import evaluation_blocks as blocks
from multimodal_jev.baseline import file_hash
from multimodal_jev.context_execution import content_key, map_io


def compare_archive(benchmark, plan, payloads, workers):
    pin = benchmark['archive_plan']
    path = Path(pin['path'])
    if file_hash(path) != pin['sha256']:
        raise ValueError('archived plan file changed')
    manifest = blocks.read(path)
    original = manifest['plan']
    if (manifest['sha256'] != content_key(original)
            or any(original[k] != plan[k] for k in ('inputs', 'policies', 'choice', 'blocks'))
            or original.get('reference_inference') != plan.get('reference_inference')):
        raise ValueError('full population, input or policy differs from the archive')
    wanted = pin.get('blocks')
    if wanted is not None and (not wanted or len(set(wanted)) != len(wanted)):
        raise ValueError('expected nonempty, distinct archive block keys')
    selected = set(wanted) if wanted is not None else {b['key'] for b in plan['blocks']}
    pairs = [(b, new) for b, new in zip(plan['blocks'], payloads, strict=True) if b['key'] in selected]
    if {b['key'] for b, _ in pairs} != selected:
        raise ValueError('archived subset contains unregistered blocks')
    archived, _ = map_io(partial(blocks.read_block, path.parent, manifest),
                         [b for b, _ in pairs], workers=workers)
    panels = {}
    for (block, new), old in zip(pairs, archived, strict=True):
        panel = panels.setdefault(block['dataset'], {}).setdefault(block['arm'],
                dict(recordings=0, pairs=0, differences=0, first_difference=None))
        for a, b in zip(old['prob']['records'], new['prob']['records'], strict=True):
            if a['id'] != b['id']:
                raise ValueError('archived recording differs')
            panel['recordings'] += 1
            panel['pairs'] += len(a['prob'])
            for index, (x, y) in enumerate(zip(a['prob'], b['prob'], strict=True)):
                if x != y:
                    panel['differences'] += 1
                    if panel['first_difference'] is None:
                        panel['first_difference'] = dict(block=block['key'], id=a['id'], column=index,
                                                         archived=x, observed=y)
    return dict(role=benchmark['role'], archive_sha256=pin['sha256'], panels=panels,
                blocks=len(pairs), differences=sum(s['differences'] for arms in panels.values() for s in arms.values()))

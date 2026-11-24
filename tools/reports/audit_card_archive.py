"""Compare every produced probability to the independently sealed full evaluation."""
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
            or any(original[k] != plan[k] for k in ('inputs', 'policies', 'choice', 'blocks'))):
        raise ValueError('full population, input or policy differs from the archive')
    archived, _ = map_io(partial(blocks.read_block, path.parent, manifest), plan['blocks'], workers=workers)
    panels = {}
    for block, old, new in zip(plan['blocks'], archived, payloads, strict=True):
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
                differences=sum(s['differences'] for arms in panels.values() for s in arms.values()))

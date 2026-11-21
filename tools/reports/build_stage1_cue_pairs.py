"""Freeze non-overlapping clean TRAIN auxiliary units, independently of model scores."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports.build_jev_boundary_panel import source_rows
from tools.reports.inventory_stage1_local_cues import local_edits
from tools.reports.jev_train_confusions import pinned


def select(rows, protected, config):
    fields = ('audio_sha256', 'speaker', 'source', 'group', 'text')
    blocked = {k: {r[k] for r in protected} for k in fields}
    pool = [r for r in rows if all(r[k] not in blocked[k] for k in fields)]
    families = config['families']
    by_label = {label: family for family in families for label in family}
    if len(by_label) != 2 * len(families) or any(len(f) != 2 for f in families):
        raise ValueError('disjoint binary confusion families required')
    lookup = defaultdict(list)
    by_id = {r['id']: r for r in pool}
    for r in pool:
        lookup[r['gold'], r['text']].append(r)
    pairs = local_edits(pool, {frozenset(f) for f in families}, config['max_changed_words'],
                        config['min_shared_words'])
    units, used = [], set()
    for p in sorted(pairs, key=lambda p: (p['kind'], p['frame'], p['references'])):
        a, b = p['labels']
        if p['kind'] == 'flip':
            # Both changed spans must explicitly carry their own family cue.
            if not all(set(e.split()) & set(by_label[l][l]) and
                       not set(e.split()) & set(by_label[l][other])
                       for e,l,other in zip(p['edits'], (a,b), (b,a))):
                continue
        elif not ('' in p['edits'] and set(p['edits']) & set(config['preserve_edge_insertions'])
                  and (p['frame'].startswith('<edit> ') or p['frame'].endswith(' <edit>'))):
            continue
        sides = [sorted(lookup[l, by_id[uid]['text']], key=lambda r:r['id'])
                 for l,uid in zip((a,b), p['references'])]
        for left in sides[0]:
            if left['audio_sha256'] in used:
                continue
            candidates = [r for r in sides[1] if r['audio_sha256'] not in used
                          and r['audio_sha256'] != left['audio_sha256']]
            if not candidates:
                continue
            right = min(candidates, key=lambda r:(r['speaker'] != left['speaker'], r['id']))
            used.update(r['audio_sha256'] for r in (left,right))
            units.append(dict(kind=p['kind'], frame=p['frame'], edits=p['edits'],
                rows=[{k:r[k] for k in ('id','audio_sha256','gold','speaker','group','source')}
                      for r in (left,right)], family=sorted(by_label[a])))
    return units, len(pool)


def build(c, provenance):
    if c['schema'] != 'stage1-cue-pairs-v1':
        raise ValueError('unknown cue configuration')
    task, rows = source_rows(c)
    root = Path(c['probe'])
    seal = json.loads(pinned(root/'MANIFEST.json', c['probe_manifest_sha256']))
    def read(name):
        return json.loads(pinned(root/name, seal['files'][name]))
    protected = read('scoring/targets.json') + list(read('support.json').values())
    by_id = {r['id']:r for r in rows}
    reserved = [by_id[r['id']] for r in protected]
    if any(r['audio_sha256'] != s['audio_sha256'] for r,s in zip(reserved,protected)):
        raise ValueError('protected probe/source mismatch')
    units, eligible = select(rows, reserved, c)
    counts = dict(Counter(p['kind'] for p in units))
    if set(counts) != {'flip','preserve'}:
        raise ValueError('both discriminative controls required')
    summary = dict(source=len(rows), protected=len(reserved), eligible_for_aux=eligible, units=counts)
    commit_json(c['output'],dict(schema=c['schema'], provenance=provenance, summary=summary,
        clean_sha256=c['clean']['sha256'],native_sha256=c['native']['sha256'],
        probe_sha256=c['probe_manifest_sha256'], candidate_ids=[x['id'] for x in task['schema']['candidates']],
        units=units))
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True); parser.add_argument('--local',required=True)
    args=parser.parse_args()
    print(json.dumps(build(*load_config(args.config,args.local)),indent=2))

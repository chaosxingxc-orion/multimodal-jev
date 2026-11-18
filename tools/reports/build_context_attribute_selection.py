"""Materialize held-out native projections using an unchanged TRAIN catalog."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from tools.reports.build_context_attribute_candidate import RELEASE_SHA, digest, targets


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_split(learning, selection, audio, menu, pairs, *, expected_count=9226):
    train_ids = {r['id'] for r in learning}
    train_groups = {r['source_group'] for r in learning}
    train_audio = {r['audio_sha256'] for r in learning}
    by_audio = {r['id']: r for r in audio}
    if (len(selection) != expected_count or len({r['id'] for r in selection}) != expected_count
            or len(by_audio) != len(audio) or set(by_audio) != {r['id'] for r in selection}):
        raise ValueError('Complete distinct selection and audio identities required')
    if set(pairs) != {e['id'] for e in menu['events']}:
        raise ValueError('TRAIN event catalog changed')
    groups = {}
    for row in selection:
        if (row['id'] in train_ids or row['source_group'] in train_groups
                or row['audio_sha256'] in train_audio):
            raise ValueError('Selection overlaps TRAIN')
        if (row['role'] != 'selection' or row['official_split'] != 'train'
                or row['known_pending_source'] is not False
                or pairs.get(row['intent']) != [row['scenario'], row['action']]
                or by_audio[row['id']]['audio_sha256'] != row['audio_sha256']):
            raise ValueError('Selection role, truth, or audio identity differs')
        if groups.setdefault(row['source_group'], row['intent']) != row['intent']:
            raise ValueError('Conflicting selection source group')
    return len(groups)


def build(release, candidate, output):
    release, candidate, output = map(Path, (release, candidate, output))
    if sha(release/'MANIFEST.json') != RELEASE_SHA:
        raise ValueError('Frozen clean release changed')
    manifest = json.loads((release/'MANIFEST.json').read_text())
    candidate_manifest = json.loads((candidate/'MANIFEST.json').read_text())
    if sha(candidate/'MANIFEST.json') != 'd98db48e619a83e7eceb308da13f257a6cc3cfa4bf701f3f17022ff99a07a482':
        raise ValueError('Pinned TRAIN candidate changed')
    for name in ('tasks.json', 'learning.jsonl', 'selection.jsonl', 'selection.audio.jsonl'):
        if sha(release/name) != manifest['files'][name]:
            raise ValueError('Release asset changed: '+name)
    if sha(candidate/'catalog.json') != candidate_manifest['files']['catalog.json']:
        raise ValueError('TRAIN catalog bytes changed')
    menu = json.loads((candidate/'catalog.json').read_text())
    if digest(menu) != candidate_manifest['catalog_sha256']:
        raise ValueError('TRAIN catalog identity changed')
    def rows(name):
        return [json.loads(s) for s in (release/name).read_text().splitlines()]
    learning, selection, audio = rows('learning.jsonl'), rows('selection.jsonl'), rows('selection.audio.jsonl')
    pairs = json.loads((release/'tasks.json').read_text())['intent']['candidate_pairs']
    group_count = validate_split(learning, selection, audio, menu, pairs)
    if any(output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve())
           for p in (release, candidate)):
        raise ValueError('Independent destination required')
    output.mkdir(exist_ok=False)
    (output/'catalog.json').write_bytes((candidate/'catalog.json').read_bytes())
    slices = Counter()
    with (output/'inputs.jsonl').open('x') as inputs, (output/'targets.jsonl').open('x') as labels:
        for row in selection:
            inp = dict(id=row['id'], audio=row['audio'], audio_sha256=row['audio_sha256'],
                       catalog_sha256=digest(menu))
            target = targets(menu, row['intent'])
            counts = [sum(v[i] == pairs[row['intent']][i] for v in pairs.values()) for i in (0, 1)]
            structural = 'both_attributes_ambiguous' if min(counts) > 1 else 'at_least_one_attribute_unique'
            slices[structural] += 1
            inputs.write(json.dumps(inp, separators=(',', ':'))+'\n')
            labels.write(json.dumps(dict(id=row['id'], input_sha256=digest(inp),
                source_group=row['source_group'], **target, structural_slice=structural,
                truth_basis=menu['supervision']), separators=(',', ':'))+'\n')
    receipt = dict(schema='context-attribute-selection-candidate-v1',
        training_dispatch_allowed=False, evaluation_dispatch_allowed=False,
        scope='Held-out native label projection; no processor/model execution or semantic process gold',
        recordings=len(selection), source_groups=group_count, decisions=len(menu['nodes']),
        structural_slices=dict(slices), catalog_sha256=digest(menu),
        release_manifest_sha256=RELEASE_SHA, train_candidate_manifest_sha256=sha(candidate/'MANIFEST.json'),
        source_files={n:sha(release/n) for n in ('learning.jsonl','selection.jsonl','selection.audio.jsonl','tasks.json')},
        files={p.name:sha(p) for p in output.iterdir()}, tool_sha256=sha(__file__),
        selection_rule='Final set-F1@0.5 descending, Final NLL ascending, earliest step',
        catalog_modified_using_selection=False, test_read=False)
    (output/'MANIFEST.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('release','candidate','output'):
        parser.add_argument('--'+key,required=True,type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.release,args.candidate,args.output),indent=2))

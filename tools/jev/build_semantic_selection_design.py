#!/usr/bin/env python3
"""Materialize current semantic selection identities, never predictions or labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from build_semantic_readout_design import compile_design

ARMS = ('Direct', 'Q_input', 'Q_credit')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def roster(selection, learning):
    for rows, role in ((selection, 'selection'), (learning, 'learning')):
        require(bool(rows) and len({r['id'] for r in rows}) == len(rows), 'unique population required')
        require(all(r['role'] == role and r['official_split'] == 'train'
                    and r['known_pending_source'] is False for r in rows), 'invalid population role')
    for field in ('id', 'source_group', 'audio_sha256'):
        require(not ({r[field] for r in selection} & {r[field] for r in learning}),
                'split overlap: ' + field)
    # Explicit allowlist: no native labels, reference text or supervision state.
    return [{k: r[k] for k in ('id', 'audio', 'audio_sha256', 'source_group')}
            for r in sorted(selection, key=lambda r: r['id'])]


def requests(population, panels):
    require(bool(panels) and len({p['panel_id'] for p in panels}) == len(panels),
            'unique panels required')
    for panel in panels:
        require(set(panel['arms']) == set(ARMS), 'current three arms required')
        require(panel['arms']['Direct'] == [] and
                panel['arms']['Q_input'] == panel['arms']['Q_credit'] == panel['process_questions'],
                'paired semantic questions required')
        require(all(q['operator'] in ('I_present', 'S_present') for q in panel['process_questions']),
                'historical membership questions forbidden')
    for row in population:
        for panel in sorted(panels, key=lambda p: p['panel_id']):
            for arm in ARMS:
                coordinate = [row['id'], panel['panel_id'], arm, 'canonical']
                identity = json.dumps(['semantic-selection-v1', *coordinate], separators=(',', ':'))
                yield dict(request_id=hashlib.sha256(identity.encode()).hexdigest(),
                           recording_id=row['id'], panel_id=panel['panel_id'], arm=arm,
                           transform='canonical')


def build(protocol_path, menu_path, readout_path, output):
    require(not output.exists(), 'output exists')
    protocol = read(protocol_path); menu = read(menu_path); readout = read(readout_path)
    require(protocol['evaluation_dispatch_allowed'] is False, 'design-only protocol required')
    release = Path(protocol['selection']['release'])
    manifest = read(release / 'MANIFEST.json')
    menu_report = read(menu['report'])
    pins = {str(menu_path): sha(menu_path), str(readout_path): sha(readout_path),
            str(release / 'MANIFEST.json'): protocol['selection']['manifest_sha256'],
            menu['report']: menu['report_sha256']}
    pins.update(menu_report['pins']); pins.update(readout['pins'])
    for name, digest in menu_report['files'].items():
        pins[str(Path(menu['report']).parent / name)] = digest
    for name in ('selection.jsonl', 'learning.jsonl', 'tasks.json'):
        pins[str(release / name)] = manifest['files'][name]
    for path, digest in pins.items():
        require(sha(path) == digest, 'changed dependency: ' + path)
    panels = read(menu['panels']); native = read(menu['native_menu'])
    expected = compile_design(panels, [n['native_id'] for n in native])
    require(all(readout[k] == v for k, v in expected.items()), 'readout/menu mismatch')
    rows = lambda name: [json.loads(line) for line in (release / name).read_text().splitlines()]
    learning = rows('learning.jsonl'); selection = rows('selection.jsonl')
    population = roster(selection, learning)
    require(len(learning) == 40513 and len(population) == protocol['selection']['recordings'] == 9226,
            'complete current population required')
    require(len({r['source_group'] for r in population}) == protocol['selection']['source_groups'] == 2146,
            'complete source grouping required')
    require(len(panels) == menu['panels_per_audio'] == 17, 'current17 panels required')
    output.mkdir(parents=True)
    # Freeze the pre-materialization protocol to avoid a circular report/config hash.
    (output / 'protocol.json').write_bytes(Path(protocol_path).read_bytes())
    pins[str(output / 'protocol.json')] = sha(output / 'protocol.json')
    (output / 'roster.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in population))
    counts = Counter()
    with (output / 'requests.jsonl').open('w') as stream:
        for request in requests(population, panels):
            stream.write(json.dumps(request, sort_keys=True) + '\n')
            counts[request['arm']] += 1
    require(dict(counts) == {arm: 156842 for arm in ARMS}, 'complete Cartesian schedule required')
    result = dict(schema='semantic-selection-evaluation-design-v1', pins=pins,
        implementation_sha256=sha(__file__), recordings=len(population), source_groups=2146,
        panels=17, requests_per_arm=dict(counts), requests_per_checkpoint=sum(counts.values()),
        transforms=['canonical'], protocol_source=str(protocol_path),
        files={name: sha(output / name) for name in ('protocol.json', 'roster.jsonl', 'requests.jsonl')},
        aggregation='Per-panel Final and equal-panel mean; no vote, union, best panel or one-call full-set claim.',
        inference_unit='One audio/panel/arm request;51 per recording per three-arm checkpoint.',
        role='internal official-TRAIN selection, not official DEV/TEST',
        training_dispatch_allowed=False, evaluation_dispatch_allowed=False, semantic_admission=0,
        scope='Identity schedule only; no audio reread, model execution, prediction, new labels, scorer or context acceptance. Diagnostic transforms require separate cost and admission.')
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, default=Path('configs/tasks/slurp/evaluation-protocol-candidate-v2.json'))
    parser.add_argument('--menu', type=Path, default=Path('configs/tasks/slurp/menu-design-candidate-v3.json'))
    parser.add_argument('--readout', type=Path, default=Path('configs/tasks/slurp/evaluation-readout-layout-candidate-v2.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.protocol, args.menu, args.readout, args.output)
    print(json.dumps({k: result[k] for k in ('recordings', 'source_groups', 'panels', 'requests_per_checkpoint')}))
